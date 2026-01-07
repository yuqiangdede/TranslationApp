from __future__ import annotations

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import asyncio
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from translation_app.audio.vad import AudioFrame, VadSegmenter
from translation_app.pipeline import FileInputThread, SegmentWorkerThread, TextTask, TextWorkerThread, UiEvent
from translation_app.shared import (
    LOCK_LANG_CHOICES,
    LOCK_LANG_CODE_TO_LABEL,
    LOCKABLE_LANG_CODES,
    asr_segment,
    STABLE_LANG_CODES,
    build_asr,
    build_hymt,
    load_json,
    translate_segment_text,
    write_json,
)
from translation_app.tts.engine import TtsConfig, TtsEngine

UiMode = Literal["mic_in", "file_in", "mic_out", "text_out"]
LangLockMode = Literal["auto", "manual"]

logger = logging.getLogger(__name__)


class _LockedAsr:
    """进程内共享 ASR：用锁串行化 transcribe，避免并发调用导致不稳定。"""

    def __init__(self, asr) -> None:  # noqa: ANN001
        self._asr = asr
        self._lock = threading.Lock()

    def transcribe(self, *, pcm16_bytes: bytes, sample_rate: int = 16000, language: str | None = None):  # noqa: ANN201
        with self._lock:
            try:
                return self._asr.transcribe(pcm16_bytes=pcm16_bytes, sample_rate=sample_rate, language=language)
            except TypeError:
                return self._asr.transcribe(pcm16_bytes=pcm16_bytes, sample_rate=sample_rate)


_GLOBAL_ASR_LOCK = threading.Lock()
_GLOBAL_ASR_KEY: tuple | None = None
_GLOBAL_ASR: _LockedAsr | None = None


def _asr_key_from_cfg(cfg: dict) -> tuple:
    asr_cfg = cfg.get("asr") if isinstance(cfg.get("asr"), dict) else {}
    return (
        str(asr_cfg.get("backend", "whisper") or "whisper").strip().lower(),
        str(asr_cfg.get("model_path", "") or "").strip(),
        str(asr_cfg.get("device", "") or "").strip().lower(),
        str(asr_cfg.get("compute_type", "") or "").strip().lower(),
        str(asr_cfg.get("cuda_bin_dir", "") or "").strip(),
        bool(asr_cfg.get("isolate_process")) if ("isolate_process" in asr_cfg) else None,
        asr_cfg.get("cuda_suspicious_min_chars_per_s"),
    )


def get_global_asr(*, cfg: dict, project_root: Path) -> _LockedAsr:
    """
    进程内全局共享 ASR（只加载一次，所有会话/上传复用）。

    注意：若你用 uvicorn 多 worker，每个 worker 进程会各自加载一次（进程隔离的正常行为）。
    """

    global _GLOBAL_ASR, _GLOBAL_ASR_KEY  # noqa: PLW0603
    key = _asr_key_from_cfg(cfg)
    with _GLOBAL_ASR_LOCK:
        if _GLOBAL_ASR is not None and _GLOBAL_ASR_KEY == key:
            return _GLOBAL_ASR
        logger.info("global ASR load: key=%s", key)
        asr, _, _, _ = build_asr(cfg, project_root=project_root)
        _GLOBAL_ASR = _LockedAsr(asr)
        _GLOBAL_ASR_KEY = key
        return _GLOBAL_ASR


def _now_ts() -> float:
    return time.time()


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


class WebSession:
    def __init__(self, *, project_root: Path, cfg_path: Path) -> None:
        self._project_root = project_root
        self._cfg_path = cfg_path
        self._cfg = load_json(cfg_path)
        self._shared_asr = get_global_asr(cfg=self._cfg, project_root=self._project_root)

        self._mode: UiMode = "file_in"
        self._lang_lock_code: str | None = None
        self._lang_lock_mode: LangLockMode = "auto"

        self._event_q: "queue.Queue[UiEvent]" = queue.Queue(maxsize=2000)

        self._text_task_q: "queue.Queue[TextTask]" = queue.Queue(maxsize=200)
        self._text_stop = threading.Event()
        self._text_thread = TextWorkerThread(task_queue=self._text_task_q, event_queue=self._event_q, stop_event=self._text_stop, cfg=self._cfg)
        self._text_thread.start()

        self._audio_stop = threading.Event()
        self._frame_q: "queue.Queue[AudioFrame]" = queue.Queue(maxsize=800)
        self._seg_q: "queue.Queue[Any]" = queue.Queue(maxsize=50)
        self._vad = VadSegmenter(sample_rate=16000, frame_ms=20)
        self._vad_thread: threading.Thread | None = None
        self._seg_worker: SegmentWorkerThread | None = None

        self._simple_active: bool = False
        self._simple_mode: UiMode = "mic_in"
        self._simple_id: int = 0
        self._simple_buf_lock = threading.Lock()
        self._simple_buf = bytearray()
        self._simple_max_bytes = 16000 * 2 * 180  # 180s @ 16kHz mono int16
        self._simple_partial_inflight = False
        self._simple_partial_last_start_ts = 0.0
        self._simple_partial_last_bytes = 0
        self._simple_partial_last_text = ""
        self._simple_partial_enabled = True
        self._simple_partial_interval_s = 0.8
        self._simple_partial_min_dur_s = 1.0
        self._simple_partial_min_step_bytes = int(0.4 * 16000 * 2)  # ~400ms
        self._simple_partial_max_dur_s = 30.0

        self._feed_buf = bytearray()
        self._cursor_samples = 0
        self._samples_per_frame = int(self._vad.sample_rate * self._vad.frame_ms / 1000)

        self._tts_replay_q: "queue.Queue[str]" = queue.Queue(maxsize=100)
        self._tts_replay_thread: threading.Thread | None = None
        self._tts_replay_engine: TtsEngine | None = None

        self._reload_ui_cfg()
        self._emit("status", {"text": "准备就绪。"})

    @property
    def cfg(self) -> dict:
        return self._cfg

    def reload_cfg(self) -> None:
        self._cfg = load_json(self._cfg_path)
        self._shared_asr = get_global_asr(cfg=self._cfg, project_root=self._project_root)
        self._restart_text_worker()
        self._reload_ui_cfg()
        if self._mode in ("mic_in", "mic_out"):
            self.stop_audio()
            self.start_audio()
        self._tts_replay_engine = None

    def save_cfg(self, data: dict) -> None:
        write_json(self._cfg_path, data)
        self._cfg = data
        self._shared_asr = get_global_asr(cfg=self._cfg, project_root=self._project_root)
        self._restart_text_worker()
        self._reload_ui_cfg()
        if self._mode in ("mic_in", "mic_out"):
            self.stop_audio()
            self.start_audio()
        self._tts_replay_engine = None

    def _restart_text_worker(self) -> None:
        try:
            self._text_stop.set()
        except Exception:
            pass
        try:
            self._text_thread.join(timeout=1.0)
        except Exception:
            pass
        self._text_stop = threading.Event()
        self._text_thread = TextWorkerThread(
            task_queue=self._text_task_q,
            event_queue=self._event_q,
            stop_event=self._text_stop,
            cfg=self._cfg,
        )
        self._text_thread.start()

    def _emit(self, kind: str, payload: dict) -> None:
        try:
            self._event_q.put_nowait(UiEvent(ts=_now_ts(), kind=kind, payload=payload))
        except queue.Full:
            pass

    def _reload_ui_cfg(self) -> None:
        ui_cfg = self._cfg.get("ui") if isinstance(self._cfg.get("ui"), dict) else {}
        partial_cfg = ui_cfg.get("simple_partial_asr") if isinstance(ui_cfg.get("simple_partial_asr"), dict) else {}

        self._simple_partial_enabled = bool(partial_cfg.get("enabled", True))
        try:
            self._simple_partial_interval_s = max(0.2, float(partial_cfg.get("interval_s", 0.8) or 0.8))
        except Exception:
            self._simple_partial_interval_s = 0.8
        try:
            self._simple_partial_min_dur_s = max(1.0, float(partial_cfg.get("min_dur_s", 1.0) or 1.0))
        except Exception:
            self._simple_partial_min_dur_s = 1.0
        try:
            step_s = float(partial_cfg.get("min_step_s", 0.4) or 0.4)
        except Exception:
            step_s = 0.4
        self._simple_partial_min_step_bytes = int(max(0.0, step_s) * 16000 * 2)
        try:
            self._simple_partial_max_dur_s = max(0.0, float(partial_cfg.get("max_dur_s", 30.0) or 30.0))
        except Exception:
            self._simple_partial_max_dur_s = 30.0

    def set_mode(self, mode: UiMode) -> None:
        prev = self._mode
        self._mode = mode
        if mode in ("mic_in", "mic_out"):
            self.start_audio()
        else:
            self.stop_audio()

        if prev != mode:
            self._emit("mode", {"mode": mode})
            self._emit("status", {"text": f"模式已切换：{mode}"})

    def _maybe_auto_switch_after_input(self, src_mode: str) -> None:
        ui_cfg = self._cfg.get("ui") if isinstance(self._cfg.get("ui"), dict) else {}
        target = str(ui_cfg.get("auto_switch_after_input", "off") or "off").strip().lower()
        if target not in ("text_out", "mic_out"):
            return
        if src_mode not in ("mic_in", "file_in"):
            return
        if self._mode != src_mode:
            return
        if not (self._lang_lock_code and self._lang_lock_code in LOCKABLE_LANG_CODES):
            return
        self.set_mode("text_out" if target == "text_out" else "mic_out")

    def set_lang_lock(self, *, code: str | None, mode: LangLockMode) -> None:
        code_norm = (code or "").strip().lower()
        if code_norm and code_norm not in LOCKABLE_LANG_CODES:
            self._emit("error", {"text": f"不支持的语种：{code_norm}"})
            return
        new_code = code_norm or None
        prev_code = self._lang_lock_code
        prev_mode = self._lang_lock_mode
        if new_code == prev_code and mode == prev_mode:
            return

        self._lang_lock_code = new_code
        self._lang_lock_mode = mode
        if self._lang_lock_code:
            if self._lang_lock_code != prev_code:
                self._emit("lock_lang", {"code": self._lang_lock_code})
        else:
            if prev_code:
                self._emit("status", {"text": "语种 L 已重置为 Auto。"})

    def start_audio(self) -> None:
        self._simple_active = False
        with self._simple_buf_lock:
            self._simple_buf = bytearray()
            self._simple_partial_inflight = False
            self._simple_partial_last_start_ts = 0.0
            self._simple_partial_last_bytes = 0
            self._simple_partial_last_text = ""
        if self._vad_thread is not None and self._vad_thread.is_alive():
            return

        self._audio_stop = threading.Event()
        self._feed_buf = bytearray()
        self._cursor_samples = 0
        while True:
            try:
                self._frame_q.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self._seg_q.get_nowait()
            except queue.Empty:
                break

        self._vad_thread = threading.Thread(
            target=self._vad.run,
            kwargs={"frame_queue": self._frame_q, "segment_queue": self._seg_q, "stop_event": self._audio_stop},
            daemon=True,
            name="WebVadThread",
        )
        self._seg_worker = SegmentWorkerThread(
            segment_queue=self._seg_q,  # type: ignore[arg-type]
            event_queue=self._event_q,  # type: ignore[arg-type]
            stop_event=self._audio_stop,
            get_mode=lambda: self._mode,
            get_lang_lock=lambda: (self._lang_lock_code, self._lang_lock_mode),
            cfg=self._cfg,
            project_root=self._project_root,
            asr=self._shared_asr,
        )
        self._vad_thread.start()
        self._seg_worker.start()

    def stop_audio(self) -> None:
        try:
            self._audio_stop.set()
        except Exception:
            pass
        for t in (self._vad_thread, self._seg_worker):
            if t is None:
                continue
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
        self._vad_thread = None
        self._seg_worker = None

    def feed_pcm16(self, pcm16_bytes: bytes) -> None:
        if self._mode not in ("mic_in", "mic_out"):
            return
        if not pcm16_bytes:
            return
        self._feed_buf.extend(pcm16_bytes)
        frame_bytes = int(self._vad.expected_frame_bytes)
        while len(self._feed_buf) >= frame_bytes:
            chunk = bytes(self._feed_buf[:frame_bytes])
            del self._feed_buf[:frame_bytes]
            ts = self._cursor_samples / float(self._vad.sample_rate)
            self._cursor_samples += self._samples_per_frame
            try:
                self._frame_q.put_nowait(AudioFrame(timestamp=ts, pcm16_bytes=chunk))
            except queue.Full:
                break

    def simple_start(self, *, mode: UiMode) -> None:
        if mode not in ("mic_in", "mic_out"):
            self._emit("error", {"text": f"simple_start 不支持的 mode={mode}。"})
            return
        with self._simple_buf_lock:
            self._simple_id += 1
            self._simple_active = True
            self._simple_mode = mode
            self._simple_buf = bytearray()
            self._simple_partial_inflight = False
            self._simple_partial_last_start_ts = 0.0
            self._simple_partial_last_bytes = 0
            self._simple_partial_last_text = ""
        self._emit("status", {"text": f"按住录音中：{mode} …"})

    def simple_feed(self, pcm16_bytes: bytes) -> None:
        if not self._simple_active:
            return
        if not pcm16_bytes:
            return
        with self._simple_buf_lock:
            if len(self._simple_buf) + len(pcm16_bytes) > self._simple_max_bytes:
                self._simple_active = False
                self._simple_buf = bytearray()
                self._emit("error", {"text": "录音过长，已停止（>180s）。"})
                return
            self._simple_buf.extend(pcm16_bytes)
            cur_buf_len = len(self._simple_buf)
            cur_id = int(self._simple_id)
            cur_mode: UiMode = self._simple_mode
        self._maybe_emit_simple_partial(mode=cur_mode, simple_id=cur_id, cur_buf_len=cur_buf_len)

    def simple_end(self) -> None:
        if not self._simple_active:
            return
        with self._simple_buf_lock:
            pcm = bytes(self._simple_buf)
            mode = self._simple_mode
            simple_id = int(self._simple_id)
            self._simple_active = False
            self._simple_buf = bytearray()
        if not pcm:
            self._emit("error", {"text": "录音为空。"})
            return
        t = threading.Thread(
            target=self._simple_process_worker,
            kwargs={"mode": mode, "pcm16_bytes": pcm, "simple_id": simple_id},
            daemon=True,
            name="WebSimpleProcess",
        )
        t.start()

    def _maybe_emit_simple_partial(self, *, mode: UiMode, simple_id: int, cur_buf_len: int) -> None:
        if not self._simple_partial_enabled:
            return
        if mode not in ("mic_in", "mic_out"):
            return

        # mic_out 需要锁定语种 L（否则最终也会报错），避免录音期间刷屏/浪费算力。
        if mode == "mic_out" and not (self._lang_lock_code and self._lang_lock_code in LOCKABLE_LANG_CODES):
            return

        dur_s = float(cur_buf_len) / (2.0 * 16000.0) if cur_buf_len else 0.0
        if dur_s < float(self._simple_partial_min_dur_s):
            return
        if float(self._simple_partial_max_dur_s) > 0.0 and dur_s > float(self._simple_partial_max_dur_s):
            return

        now = _now_ts()
        with self._simple_buf_lock:
            if not self._simple_active:
                return
            if int(self._simple_id) != int(simple_id):
                return
            if self._simple_partial_inflight:
                return
            if (now - float(self._simple_partial_last_start_ts)) < float(self._simple_partial_interval_s):
                return
            if (cur_buf_len - int(self._simple_partial_last_bytes)) < int(self._simple_partial_min_step_bytes):
                return
            pcm = bytes(self._simple_buf)
            self._simple_partial_inflight = True
            self._simple_partial_last_start_ts = now
            self._simple_partial_last_bytes = cur_buf_len

        t = threading.Thread(
            target=self._simple_partial_worker,
            kwargs={"mode": mode, "simple_id": simple_id, "pcm16_bytes": pcm},
            daemon=True,
            name="WebSimplePartialAsr",
        )
        t.start()

    def _simple_partial_worker(self, *, mode: UiMode, simple_id: int, pcm16_bytes: bytes) -> None:
        try:
            locked, lock_mode = (self._lang_lock_code, self._lang_lock_mode)
            if mode == "mic_out" and not (locked and locked in LOCKABLE_LANG_CODES):
                return

            lang_cfg = self._cfg.get("language") if isinstance(self._cfg.get("language"), dict) else {}
            thr = float(lang_cfg.get("prob_threshold", 0.7) or 0.7)

            asr_out = asr_segment(
                pcm16_bytes=pcm16_bytes,
                sample_rate=16000,
                asr=self._shared_asr,
                direction=("to_zh" if mode == "mic_in" else "to_locked"),
                locked_lang_code=locked,
                lock_mode=lock_mode,
                lock_prob_threshold=thr,
            )
            if not asr_out:
                return

            with self._simple_buf_lock:
                if not self._simple_active:
                    return
                if int(self._simple_id) != int(simple_id):
                    return

            text_in = str(asr_out.get("text_in", "") or "")
            with self._simple_buf_lock:
                if text_in == str(self._simple_partial_last_text or ""):
                    return
                self._simple_partial_last_text = text_in

            dur_s = float(len(pcm16_bytes)) / (2.0 * 16000.0) if pcm16_bytes else 0.0
            self._emit(
                "asr_partial",
                {
                    "mode": mode,
                    "simple_id": simple_id,
                    "seg_id": 1,
                    "seg_start_s": 0.0,
                    "seg_end_s": dur_s,
                    "lang_in": asr_out.get("lang_in", ""),
                    "prob": float(asr_out.get("prob", 0.0) or 0.0),
                    "text_in": text_in,
                    "t_asr_ms": float(asr_out.get("t_asr_ms", 0.0) or 0.0),
                },
            )
        except Exception:
            # partial 仅用于预览，不应影响主流程
            return
        finally:
            with self._simple_buf_lock:
                self._simple_partial_inflight = False

    def _simple_process_worker(self, *, mode: UiMode, pcm16_bytes: bytes, simple_id: int) -> None:
        try:
            hymt = build_hymt(self._cfg)
            locked, lock_mode = (self._lang_lock_code, self._lang_lock_mode)
            if mode == "mic_out" and not (locked and locked in LOCKABLE_LANG_CODES):
                self._emit("error", {"text": "尚未锁定语种 L。"})
                return

            lang_cfg = self._cfg.get("language") if isinstance(self._cfg.get("language"), dict) else {}
            thr = float(lang_cfg.get("prob_threshold", 0.7) or 0.7)

            asr_out = asr_segment(
                pcm16_bytes=pcm16_bytes,
                sample_rate=16000,
                asr=self._shared_asr,
                direction=("to_zh" if mode == "mic_in" else "to_locked"),
                locked_lang_code=locked,
                lock_mode=lock_mode,
                lock_prob_threshold=thr,
            )
            if not asr_out:
                self._emit("error", {"text": "未识别到有效语音（ASR 输出为空）。"})
                return
            if asr_out.get("suggested_lock_code"):
                self._emit("lock_lang", {"code": asr_out["suggested_lock_code"]})

            dur_s = float(len(pcm16_bytes)) / (2.0 * 16000.0) if pcm16_bytes else 0.0
            self._emit(
                "asr",
                {
                    "mode": mode,
                    "simple_id": simple_id,
                    "seg_id": 1,
                    "seg_start_s": 0.0,
                    "seg_end_s": dur_s,
                    "lang_in": asr_out.get("lang_in", ""),
                    "prob": float(asr_out.get("prob", 0.0) or 0.0),
                    "text_in": asr_out.get("text_in", ""),
                    "t_asr_ms": float(asr_out.get("t_asr_ms", 0.0) or 0.0),
                },
            )

            tr_out = translate_segment_text(
                text_in=str(asr_out.get("text_in", "") or ""),
                lang_in=str(asr_out.get("lang_in", "") or ""),
                hymt=hymt,
                direction=("to_zh" if mode == "mic_in" else "to_locked"),
                locked_lang_code=locked,
            )
            self._emit(
                "tr",
                {
                    "mode": mode,
                    "simple_id": simple_id,
                    "seg_id": 1,
                    "seg_start_s": 0.0,
                    "seg_end_s": dur_s,
                    "dst_code": tr_out.get("dst_code", ""),
                    "text_out": tr_out.get("text_out", ""),
                    "t_tr_ms": float(tr_out.get("t_tr_ms", 0.0) or 0.0),
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._emit("error", {"text": f"处理异常：{exc}"})

    def submit_text(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if not (self._lang_lock_code and self._lang_lock_code in LOCKABLE_LANG_CODES):
            self._emit("error", {"text": "尚未锁定语种 L。"})
            return
        try:
            self._text_task_q.put_nowait(TextTask(text_zh=text, target_code=self._lang_lock_code))
        except queue.Full:
            self._emit("error", {"text": "任务队列已满。"})

    def request_tts(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return

        tts_cfg = self._cfg.get("tts", {}) if isinstance(self._cfg.get("tts", {}), dict) else {}
        enabled = bool(tts_cfg.get("enabled", False))
        backend = str(tts_cfg.get("backend", "auto") or "auto")
        if not enabled:
            self._emit("error", {"text": "TTS 未启用。"})
            return

        if self._tts_replay_engine is None:
            self._tts_replay_engine = TtsEngine(TtsConfig(backend=backend))
        if self._tts_replay_engine.backend == "disabled":
            self._emit("error", {"text": "TTS 不可用。"})
            return

        if self._tts_replay_thread is None or not self._tts_replay_thread.is_alive():
            self._tts_replay_thread = threading.Thread(target=self._tts_replay_worker, daemon=True, name="WebTtsReplayThread")
            self._tts_replay_thread.start()

        try:
            self._tts_replay_q.put_nowait(text)
        except queue.Full:
            pass

    def _tts_replay_worker(self) -> None:
        while True:
            try:
                text = self._tts_replay_q.get()
            except Exception:
                continue
            # 若连续点击“播放”，只保留最新一次请求，并打断当前播放。
            try:
                while True:
                    text = self._tts_replay_q.get_nowait()
            except queue.Empty:
                pass
            try:
                if self._tts_replay_engine is not None:
                    self._tts_replay_engine.stop()
                    self._tts_replay_engine.speak_async(text)
            except Exception:
                pass

    def close(self) -> None:
        self.stop_audio()
        try:
            self._text_stop.set()
        except Exception:
            pass
        try:
            self._text_thread.join(timeout=1.0)
        except Exception:
            pass

    def drain_events(self, *, timeout: float = 0.2) -> dict | None:
        try:
            ev = self._event_q.get(timeout=timeout)
        except queue.Empty:
            return None
        if ev.kind == "lock_lang":
            code = str((ev.payload or {}).get("code", "") or "").strip().lower()
            if code and code in LOCKABLE_LANG_CODES and self._lang_lock_mode == "auto":
                self._lang_lock_code = code
        if ev.kind in ("segment", "tr"):
            try:
                src_mode = str((ev.payload or {}).get("mode", "") or "")
            except Exception:
                src_mode = ""
            if src_mode:
                self._maybe_auto_switch_after_input(src_mode)
        return {"ts": float(ev.ts), "kind": str(ev.kind), "payload": dict(ev.payload or {})}


def _index_html() -> str:
    root = Path(__file__).resolve().parent
    html_path = root / "static" / "index.html"
    return html_path.read_text(encoding="utf-8")


def create_app() -> FastAPI:
    app = FastAPI(title="Translation Web UI")

    project_root = Path(__file__).resolve().parents[2]
    cfg_path = (project_root / "config.json").resolve()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        return HTMLResponse(_index_html(), headers={"Cache-Control": "no-store"})

    @app.head("/")
    async def head_root() -> Response:
        return Response()

    @app.get("/static/{path:path}")
    async def static_file(path: str):
        full = Path(__file__).resolve().parent / "static" / path
        if not full.is_file():
            return HTMLResponse("Not found", status_code=404)
        return FileResponse(full, headers={"Cache-Control": "no-store"})

    @app.get("/api/languages")
    async def languages() -> dict:
        return {
            "choices": [
                {"code": c, "label": LOCK_LANG_CODE_TO_LABEL.get(c, c), "stable": (c in STABLE_LANG_CODES)} for c, _ in LOCK_LANG_CHOICES
            ]
        }

    @app.get("/api/config")
    async def get_config() -> dict:
        return {"config": load_json(cfg_path)}

    @app.post("/api/config")
    async def set_config(data: dict) -> dict:
        if not isinstance(data, dict):
            return {"ok": False, "error": "config must be a JSON object"}
        write_json(cfg_path, data)
        return {"ok": True}

    @app.post("/api/file_in")
    async def file_in(file: UploadFile = File(...)) -> dict:
        # Process file synchronously and return events list.
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in (".wav", ".mp3"):
            return {"ok": False, "error": "仅支持 WAV/MP3 文件。"}

        tmp_dir = project_root / "res" / "_web_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{int(time.time()*1000)}_{Path(file.filename or 'upload').name}"
        data = await file.read()
        tmp_path.write_bytes(data)

        ev_q: "queue.Queue[UiEvent]" = queue.Queue(maxsize=2000)
        stop = threading.Event()
        cfg = load_json(cfg_path)
        shared_asr = get_global_asr(cfg=cfg, project_root=project_root)
        t = FileInputThread(
            wav_path=tmp_path,
            event_queue=ev_q,
            stop_event=stop,
            cfg=cfg,
            project_root=project_root,
            asr=shared_asr,
        )
        t.run()

        events: list[dict[str, Any]] = []
        while True:
            try:
                ev = ev_q.get_nowait()
                events.append({"ts": float(ev.ts), "kind": str(ev.kind), "payload": dict(ev.payload or {})})
            except queue.Empty:
                break

        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[call-arg]
        except Exception:
            pass

        return {"ok": True, "events": events}

    @app.post("/api/file_in_stream")
    async def file_in_stream(file: UploadFile = File(...)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in (".wav", ".mp3"):
            return {"ok": False, "error": "仅支持 WAV/MP3 文件。"}

        tmp_dir = project_root / "res" / "_web_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{int(time.time()*1000)}_{Path(file.filename or 'upload').name}"
        data = await file.read()
        tmp_path.write_bytes(data)

        ev_q: "queue.Queue[UiEvent]" = queue.Queue(maxsize=2000)
        stop = threading.Event()
        cfg = load_json(cfg_path)
        shared_asr = get_global_asr(cfg=cfg, project_root=project_root)
        t = FileInputThread(
            wav_path=tmp_path,
            event_queue=ev_q,
            stop_event=stop,
            cfg=cfg,
            project_root=project_root,
            asr=shared_asr,
        )
        t.start()

        async def gen():
            try:
                while t.is_alive() or (not ev_q.empty()):
                    try:
                        ev = ev_q.get(timeout=0.2)
                    except queue.Empty:
                        await asyncio.sleep(0.05)
                        continue
                    yield (json.dumps({"ts": float(ev.ts), "kind": str(ev.kind), "payload": dict(ev.payload or {})}, ensure_ascii=False) + "\n").encode(
                        "utf-8"
                    )
                try:
                    t.join(timeout=0.2)
                except Exception:
                    pass
            except asyncio.CancelledError:
                try:
                    stop.set()
                except Exception:
                    pass
                raise
            finally:
                try:
                    stop.set()
                except Exception:
                    pass
                try:
                    t.join(timeout=0.5)
                except Exception:
                    pass
                try:
                    tmp_path.unlink(missing_ok=True)  # type: ignore[call-arg]
                except Exception:
                    pass

        return StreamingResponse(gen(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        session = WebSession(project_root=project_root, cfg_path=cfg_path)
        loop = asyncio.get_running_loop()
        stop_sender = threading.Event()

        def sender() -> None:
            while not stop_sender.is_set():
                ev = session.drain_events(timeout=0.2)
                if ev is None:
                    continue
                try:
                    fut = asyncio.run_coroutine_threadsafe(ws.send_json({"type": "event", **ev}), loop)
                    fut.result(timeout=2.0)
                except Exception:
                    break

        sender_thread = threading.Thread(target=sender, daemon=True, name="WebWsSender")
        sender_thread.start()

        try:
            await ws.send_json(
                {
                    "type": "hello",
                    "mode": session._mode,
                    "lock": {"code": session._lang_lock_code, "mode": session._lang_lock_mode},
                }
            )
            while True:
                try:
                    msg = await ws.receive()
                except (WebSocketDisconnect, RuntimeError):
                    break
                if msg.get("type") == "websocket.disconnect":
                    break
                if "bytes" in msg and msg["bytes"] is not None:
                    if session._simple_active:
                        session.simple_feed(bytes(msg["bytes"]))
                    else:
                        session.feed_pcm16(bytes(msg["bytes"]))
                    continue
                text = msg.get("text")
                if not text:
                    continue
                try:
                    data = json.loads(text)
                except Exception:
                    continue
                mtype = str(data.get("type", "") or "")
                if mtype == "set_mode":
                    logger.info("ws recv: set_mode mode=%s", str(data.get("mode", "") or ""))
                elif mtype == "set_lock":
                    logger.info("ws recv: set_lock code=%s mode=%s", str(data.get("code", "") or ""), str(data.get("lock_mode", "") or ""))
                elif mtype == "simple_start":
                    logger.info("ws recv: simple_start mode=%s", str(data.get("mode", "") or ""))
                elif mtype == "simple_end":
                    logger.info("ws recv: simple_end")
                elif mtype == "text_out":
                    logger.info("ws recv: text_out len=%d", len(str(data.get("text", "") or "")))
                elif mtype == "tts":
                    logger.info("ws recv: tts len=%d", len(str(data.get("text", "") or "")))
                elif mtype == "reload_config":
                    logger.info("ws recv: reload_config")
                if mtype == "set_mode":
                    mode = str(data.get("mode", "") or "")
                    if mode in ("mic_in", "file_in", "mic_out", "text_out"):
                        session.set_mode(mode)  # type: ignore[arg-type]
                elif mtype == "simple_start":
                    mode = str(data.get("mode", "") or "")
                    if mode in ("mic_in", "mic_out"):
                        session.simple_start(mode=mode)  # type: ignore[arg-type]
                elif mtype == "simple_end":
                    session.simple_end()
                elif mtype == "set_lock":
                    code = data.get("code")
                    lmode = str(data.get("lock_mode", "manual") or "manual").strip().lower()
                    session.set_lang_lock(code=str(code) if code else None, mode=("auto" if lmode == "auto" else "manual"))
                elif mtype == "text_out":
                    session.submit_text(str(data.get("text", "") or ""))
                elif mtype == "tts":
                    session.request_tts(str(data.get("text", "") or ""))
                elif mtype == "reload_config":
                    session.reload_cfg()
                    session._emit("status", {"text": "配置已重新加载。"})
        except WebSocketDisconnect:
            pass
        finally:
            stop_sender.set()
            session.close()
            try:
                sender_thread.join(timeout=0.5)
            except Exception:
                pass

    return app


def main() -> int:
    import argparse
    import ssl

    import uvicorn

    _configure_logging()

    parser = argparse.ArgumentParser(description="Translation Web UI (FastAPI)")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP/HTTPS bind host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--https-port", type=int, default=443, help="HTTPS port (default 443)")
    parser.add_argument("--ssl-certfile", default="", help="Path to SSL certificate PEM for HTTPS")
    parser.add_argument("--ssl-keyfile", default="", help="Path to SSL private key PEM for HTTPS")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    app = create_app()

    ssl_certfile = str(getattr(args, "ssl_certfile", "") or "").strip()
    ssl_keyfile = str(getattr(args, "ssl_keyfile", "") or "").strip()
    enable_https = bool(ssl_certfile and ssl_keyfile)

    # uvicorn --reload 在多 server 模式下不可控，这里直接降级为只启动 HTTP。
    if args.reload and enable_https:
        logger.warning("--reload 与 HTTPS 双端口模式不兼容，将只启动 HTTP（不启 HTTPS）。")
        enable_https = False

    servers: list[uvicorn.Server] = []
    threads: list[threading.Thread] = []

    def start_server_in_thread(config: uvicorn.Config, name: str) -> uvicorn.Server:
        server = uvicorn.Server(config)

        def _run() -> None:
            try:
                server.run()
            except Exception as exc:  # noqa: BLE001
                logger.error("%s server stopped: %s", name, exc)

        t = threading.Thread(target=_run, daemon=True, name=name)
        t.start()
        servers.append(server)
        threads.append(t)
        return server

    http_cfg = uvicorn.Config(
        app,
        host=str(args.host),
        port=int(args.port),
        reload=bool(args.reload),
        log_level="info",
        log_config=None,
    )

    if enable_https:
        # 预检 SSL 文件可读性（避免启动后悄悄失败）
        try:
            _ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            _ctx.load_cert_chain(certfile=ssl_certfile, keyfile=ssl_keyfile)
        except Exception as exc:  # noqa: BLE001
            logger.error("HTTPS 证书/私钥加载失败，跳过 HTTPS：%s", exc)
            enable_https = False

    if enable_https:
        https_cfg = uvicorn.Config(
            app,
            host=str(args.host),
            port=int(args.https_port),
            reload=False,
            log_level="info",
            log_config=None,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
        )
        try:
            start_server_in_thread(https_cfg, "UvicornHTTPS")
            logger.info("HTTPS enabled: https://%s:%d", str(args.host), int(args.https_port))
        except OSError as exc:
            logger.error("HTTPS 启动失败（可能端口被占用或无权限），跳过 HTTPS：%s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.error("HTTPS 启动失败，跳过 HTTPS：%s", exc)

    # HTTP 放在主线程跑，便于 Ctrl+C 停止
    try:
        uvicorn.Server(http_cfg).run()
    finally:
        for s in servers:
            try:
                s.should_exit = True
            except Exception:
                pass
        for t in threads:
            try:
                t.join(timeout=0.5)
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

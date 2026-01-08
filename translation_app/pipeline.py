"""运行管线（线程/任务）：供 Web 服务复用。"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from translation_app.audio.vad import AudioFrame, SpeechSegment, VadSegmenter
from translation_app.shared import (
    LOCKABLE_LANG_CODES,
    asr_segment,
    build_asr,
    build_asr_corrector,
    correct_asr_text,
    process_text,
    translate_segment_text,
)
from translation_app.translate.hymt import HyMtClient, HyMtConfig, HyMtError
from translation_app.tts.engine import TtsConfig, TtsEngine

logger = logging.getLogger(__name__)

UiMode = Literal["mic_in", "file_in", "mic_out", "text_out"]
LangLockMode = Literal["auto", "manual"]


@dataclass(frozen=True)
class UiEvent:
    ts: float
    kind: Literal["status", "error", "asr", "asr_partial", "tr", "segment", "lock_lang", "file_done", "mode"]
    payload: dict


@dataclass(frozen=True)
class TextTask:
    text_zh: str
    target_code: str


def _now_ts() -> float:
    return time.time()


def _resample_linear(x, *, in_sr: int, out_sr: int):
    import numpy as np

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if in_sr == out_sr:
        return x
    if x.size <= 1:
        return np.zeros((int(round(x.size * (out_sr / float(in_sr)))),), dtype=np.float32)
    out_len = int(round(x.size * (out_sr / float(in_sr))))
    if out_len <= 1:
        return np.zeros((1,), dtype=np.float32)
    xp = np.linspace(0.0, float(x.size - 1), num=out_len, dtype=np.float32)
    xi = np.arange(x.size, dtype=np.float32)
    return np.interp(xp, xi, x).astype(np.float32, copy=False)


def _load_wav_pcm16_16k_mono(path: Path) -> bytes:
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as wf:
        ch = int(wf.getnchannels())
        sr = int(wf.getframerate())
        sw = int(wf.getsampwidth())
        n = int(wf.getnframes())
        raw = wf.readframes(n)

    if sw != 2:
        raise RuntimeError(f"仅支持 16-bit PCM WAV（sampwidth=2），当前={sw}")
    if ch <= 0:
        raise RuntimeError(f"WAV 声道数异常：{ch}")

    x = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    x = x.astype(np.float32) / 32768.0
    if sr != 16000:
        x = _resample_linear(x, in_sr=sr, out_sr=16000)
    pcm16 = np.clip(x * 32767.0, -32768.0, 32767.0).astype(np.int16)
    return pcm16.tobytes()


def _load_audio_pcm16_16k_mono(path: Path) -> bytes:
    suf = (path.suffix or "").lower()
    if suf == ".wav":
        return _load_wav_pcm16_16k_mono(path)
    if suf != ".mp3":
        raise RuntimeError(f"仅支持音频格式：.wav / .mp3（当前={path.suffix or '无后缀'}）")

    def _load_mp3_via_pyav() -> bytes:
        try:
            import av  # PyAV（当前项目环境已包含）
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("MP3 解码失败：缺少 PyAV 依赖（av）。") from exc

        try:
            container = av.open(str(path))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"MP3 打开失败：{path.name}") from exc

        try:
            audio_stream = None
            for s in container.streams:
                if getattr(s, "type", None) == "audio":
                    audio_stream = s
                    break
            if audio_stream is None:
                raise RuntimeError("文件中未找到音频轨道。")

            resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
            chunks: list[bytes] = []

            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    out = resampler.resample(frame)
                    out_frames = out if isinstance(out, list) else [out]
                    for f in out_frames:
                        if f is None:
                            continue
                        arr = f.to_ndarray()
                        chunks.append(arr.tobytes())

            try:
                out = resampler.resample(None)
                out_frames = out if isinstance(out, list) else [out]
                for f in out_frames:
                    if f is None:
                        continue
                    arr = f.to_ndarray()
                    chunks.append(arr.tobytes())
            except Exception:
                pass

            pcm16 = b"".join(chunks)
            if not pcm16:
                raise RuntimeError("未解码出有效音频数据。")
            return pcm16
        finally:
            try:
                container.close()
            except Exception:
                pass

    # PyAV 路径更轻量（无需 torch/torchaudio），也更稳定地覆盖 Windows 的 MP3 解码。
    return _load_mp3_via_pyav()


def _segment_pcm16_with_vad(*, vad: VadSegmenter, pcm16_bytes: bytes) -> list[SpeechSegment]:
    frame_bytes = int(vad.expected_frame_bytes)
    if frame_bytes <= 0:
        return []

    segments: list[SpeechSegment] = []
    segment_id = 0
    sample_cursor = 0
    samples_per_frame = int(vad.sample_rate * vad.frame_ms / 1000)

    pre_buffer: list[AudioFrame] = []
    in_segment = False
    speech_candidate = 0
    silence_count = 0
    last_speech_index = -1
    frames: list[AudioFrame] = []

    def emit_segment(effective_frames: list[AudioFrame]) -> None:
        nonlocal segment_id
        if len(effective_frames) < int(vad.min_frames):
            return
        segment_id += 1
        start_time = effective_frames[0].timestamp
        end_time = effective_frames[-1].timestamp + (vad.frame_ms / 1000.0)
        pcm = b"".join(f.pcm16_bytes for f in effective_frames)
        segments.append(
            SpeechSegment(
                segment_id=segment_id,
                start_time=start_time,
                end_time=end_time,
                sample_rate=vad.sample_rate,
                pcm16_bytes=pcm,
            )
        )

    for off in range(0, len(pcm16_bytes) - frame_bytes + 1, frame_bytes):
        chunk = pcm16_bytes[off : off + frame_bytes]
        ts = sample_cursor / float(vad.sample_rate)
        sample_cursor += samples_per_frame
        frame = AudioFrame(timestamp=ts, pcm16_bytes=chunk)

        try:
            is_speech = vad.vad.is_speech(chunk, vad.sample_rate)
        except Exception:
            continue

        if not in_segment:
            pre_buffer.append(frame)
            if len(pre_buffer) > int(vad.pre_speech_frames):
                pre_buffer = pre_buffer[-int(vad.pre_speech_frames) :]
            if is_speech:
                speech_candidate += 1
                if speech_candidate >= int(vad.start_trigger_frames):
                    in_segment = True
                    frames = list(pre_buffer)
                    last_speech_index = len(frames) - 1
                    silence_count = 0
                    speech_candidate = 0
            else:
                speech_candidate = 0
            continue

        frames.append(frame)
        if is_speech:
            silence_count = 0
            last_speech_index = len(frames) - 1
            continue

        silence_count += 1
        if silence_count < int(vad.silence_frames_needed):
            continue

        effective_end = 0 if last_speech_index < 0 else min(len(frames), last_speech_index + 1 + int(vad.tail_keep_frames))
        effective_frames = frames[:effective_end] if effective_end > 0 else []
        emit_segment(effective_frames)

        pre_buffer = frames[-int(vad.pre_speech_frames) :] if int(vad.pre_speech_frames) > 0 else []
        in_segment = False
        frames = []
        silence_count = 0
        last_speech_index = -1
        speech_candidate = 0

    if in_segment and frames:
        effective_end = 0 if last_speech_index < 0 else min(len(frames), last_speech_index + 1 + int(vad.tail_keep_frames))
        effective_frames = frames[:effective_end] if effective_end > 0 else []
        emit_segment(effective_frames)

    return segments


class SegmentWorkerThread(threading.Thread):
    """消费 SpeechSegment：ASR + 翻译 +（可选）TTS。"""

    def __init__(
        self,
        *,
        segment_queue: "queue.Queue[SpeechSegment]",
        event_queue: "queue.Queue[UiEvent]",
        stop_event: threading.Event,
        get_mode: Callable[[], UiMode],
        get_lang_lock: Callable[[], tuple[str | None, LangLockMode]],
        cfg: dict,
        project_root: Path,
        asr=None,
        hymt: HyMtClient | None = None,
        tts: TtsEngine | None = None,
    ) -> None:
        super().__init__(daemon=True, name="SegmentWorkerThread")
        self._seg_q = segment_queue
        self._ev_q = event_queue
        self._stop_event = stop_event
        self._get_mode = get_mode
        self._get_lock = get_lang_lock
        self._cfg = cfg
        self._project_root = project_root

        self._asr = asr
        self._hymt: HyMtClient | None = hymt
        self._tts: TtsEngine | None = tts
        self._corrector = None
        self._models_ready = False

        lang_cfg = cfg.get("language") if isinstance(cfg.get("language"), dict) else {}
        self._thr = float(lang_cfg.get("prob_threshold", 0.7) or 0.7)

        tts_cfg = cfg.get("tts") if isinstance(cfg.get("tts"), dict) else {}
        self._tts_enabled = bool(tts_cfg.get("enabled", True))

    def _put(self, kind: str, payload: dict) -> None:
        try:
            self._ev_q.put_nowait(UiEvent(ts=_now_ts(), kind=kind, payload=payload))
        except queue.Full:
            pass

    def _init_models(self) -> None:
        if self._models_ready:
            return
        self._asr, _, _, _ = build_asr(self._cfg, project_root=self._project_root)
        self._hymt = HyMtClient(HyMtConfig(api_url=(self._cfg.get("translate") or {}).get("api_url", "http://localhost:1234/v1/chat/completions")))
        self._tts = TtsEngine(TtsConfig(backend=("disabled" if not self._tts_enabled else ((self._cfg.get("tts") or {}).get("backend", "auto")))))
        self._corrector = build_asr_corrector(self._cfg)
        self._models_ready = True

    def run(self) -> None:
        try:
            self._put("status", {"text": "初始化 ASR/翻译/TTS ..."})
            self._init_models()
            self._put("status", {"text": "音频管线已就绪。"})
        except Exception as exc:  # noqa: BLE001
            self._put("error", {"text": f"初始化失败：{exc}"})
            self._stop_event.set()
            return

        while not self._stop_event.is_set():
            try:
                seg = self._seg_q.get(timeout=0.2)
            except queue.Empty:
                continue

            mode = self._get_mode()
            if mode == "text_out":
                continue

            locked, lock_mode = self._get_lock()
            try:
                should_tts = bool(self._tts_enabled and mode == "mic_out" and (self._tts is not None))
                asr_out = asr_segment(
                    pcm16_bytes=seg.pcm16_bytes,
                    sample_rate=seg.sample_rate,
                    asr=self._asr,  # type: ignore[arg-type]
                    direction=("to_zh" if mode in ("mic_in", "file_in") else "to_locked"),
                    locked_lang_code=locked,
                    lock_mode=lock_mode,
                    lock_prob_threshold=self._thr,
                )
                if not asr_out:
                    continue
                lang_in = str(asr_out.get("lang_in", "") or "").strip().lower()
                text_in_raw = str(asr_out.get("text_in", "") or "").strip()
                if text_in_raw == "[无有效内容]":
                    continue
                text_in = text_in_raw
                include_raw = False
                if mode in ("mic_in", "mic_out") and self._corrector is not None:
                    include_raw = True
                    if mode == "mic_out" and lang_in == "zh":
                        text_in = correct_asr_text(text_in=text_in_raw, corrector=self._corrector)
                        if not text_in:
                            continue
                asr_out["text_in"] = text_in

                if asr_out.get("suggested_lock_code"):
                    self._put("lock_lang", {"code": asr_out["suggested_lock_code"]})
                payload = {
                    "mode": mode,
                    "seg_id": int(getattr(seg, "segment_id", 0) or 0),
                    "seg_start_s": float(getattr(seg, "start_time", 0.0) or 0.0),
                    "seg_end_s": float(getattr(seg, "end_time", 0.0) or 0.0),
                    "lang_in": asr_out.get("lang_in", ""),
                    "prob": float(asr_out.get("prob", 0.0) or 0.0),
                    "text_in": text_in,
                    "t_asr_ms": float(asr_out.get("t_asr_ms", 0.0) or 0.0),
                }
                if include_raw:
                    payload["text_in_raw"] = text_in_raw
                self._put("asr", payload)

                tr_out = translate_segment_text(
                    text_in=text_in,
                    lang_in=str(asr_out.get("lang_in", "") or ""),
                    hymt=self._hymt,  # type: ignore[arg-type]
                    direction=("to_zh" if mode in ("mic_in", "file_in") else "to_locked"),
                    locked_lang_code=locked,
                )
                self._put(
                    "tr",
                    {
                        "mode": mode,
                        "seg_id": int(getattr(seg, "segment_id", 0) or 0),
                        "seg_start_s": float(getattr(seg, "start_time", 0.0) or 0.0),
                        "seg_end_s": float(getattr(seg, "end_time", 0.0) or 0.0),
                        "dst_code": tr_out.get("dst_code", ""),
                        "text_out": tr_out.get("text_out", ""),
                        "t_tr_ms": float(tr_out.get("t_tr_ms", 0.0) or 0.0),
                    },
                )

                # TTS 放到 UI 文本输出之后执行（仍在后台线程中，不阻塞主线程）
                if should_tts:
                    try:
                        out_text = str(tr_out.get("text_out", "") or "")
                        if out_text.strip():
                            self._tts.speak(out_text)  # type: ignore[union-attr]
                    except Exception:
                        pass
            except HyMtError as exc:
                self._put("error", {"text": f"翻译失败：{exc}"})
            except Exception as exc:  # noqa: BLE001
                self._put("error", {"text": f"处理异常：{exc}"})


class TextWorkerThread(threading.Thread):
    """消费 TextTask：中文 -> 语种 L（可选 TTS）。"""

    def __init__(
        self,
        *,
        task_queue: "queue.Queue[TextTask]",
        event_queue: "queue.Queue[UiEvent]",
        stop_event: threading.Event,
        cfg: dict,
    ) -> None:
        super().__init__(daemon=True, name="TextWorkerThread")
        self._task_q = task_queue
        self._ev_q = event_queue
        self._stop_event = stop_event

        translate_cfg = cfg.get("translate") if isinstance(cfg.get("translate"), dict) else {}
        tts_cfg = cfg.get("tts") if isinstance(cfg.get("tts"), dict) else {}
        self._tts_enabled = bool(tts_cfg.get("enabled", True))

        self._hymt = HyMtClient(HyMtConfig(api_url=str(translate_cfg.get("api_url", "http://localhost:1234/v1/chat/completions"))))
        self._tts = TtsEngine(TtsConfig(backend=("disabled" if not self._tts_enabled else str(tts_cfg.get("backend", "auto")))))

    def _put(self, kind: str, payload: dict) -> None:
        try:
            self._ev_q.put_nowait(UiEvent(ts=_now_ts(), kind=kind, payload=payload))
        except queue.Full:
            pass

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._task_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                should_tts = bool(self._tts_enabled and (self._tts is not None))
                res = process_text(
                    text_zh=task.text_zh,
                    hymt=self._hymt,
                    tts=None,  # 先把文本输出到 UI，再执行 TTS
                    locked_lang_code=task.target_code,
                    tts_enabled=False,
                )
                if not res:
                    continue
                self._put(
                    "segment",
                    {
                        "mode": "text_out",
                        "lang_in": res.get("lang_in", "zh"),
                        "prob": float(res.get("prob", 1.0) or 1.0),
                        "text_in": res.get("text_in", ""),
                        "dst_code": res.get("dst_code", ""),
                        "text_out": res.get("text_out", ""),
                        "tts": bool(res.get("tts_played", False)),
                        "t_asr_ms": 0.0,
                        "t_tr_ms": float(res.get("t_tr_ms", 0.0) or 0.0),
                        "t_tts_ms": float(res.get("t_tts_ms", 0.0) or 0.0),
                    },
                )
                if should_tts:
                    try:
                        out_text = str(res.get("text_out", "") or "")
                        if out_text.strip():
                            self._tts.speak(out_text)
                    except Exception:
                        pass
            except HyMtError as exc:
                self._put("error", {"text": f"翻译失败：{exc}"})
            except Exception as exc:  # noqa: BLE001
                self._put("error", {"text": f"处理异常：{exc}"})


class FileInputThread(threading.Thread):
    """文件输入：音频文件（WAV/MP3）-> ASR -> 翻译为中文（并尝试锁定语种 L）。"""

    def __init__(
        self,
        *,
        wav_path: Path,
        event_queue: "queue.Queue[UiEvent]",
        stop_event: threading.Event,
        cfg: dict,
        project_root: Path,
        asr=None,
        hymt: HyMtClient | None = None,
    ) -> None:
        super().__init__(daemon=True, name="FileInputThread")
        self._path = wav_path
        self._ev_q = event_queue
        self._stop_event = stop_event
        self._cfg = cfg
        self._project_root = project_root
        self._asr = asr
        self._hymt = hymt

        lang_cfg = cfg.get("language") if isinstance(cfg.get("language"), dict) else {}
        self._prob_thr = float(lang_cfg.get("prob_threshold", 0.7) or 0.7)

    def _put(self, kind: str, payload: dict) -> None:
        try:
            self._ev_q.put_nowait(UiEvent(ts=_now_ts(), kind=kind, payload=payload))
        except queue.Full:
            pass

    def run(self) -> None:
        try:
            show_name = f"{self._path.parent.name}\\{self._path.name}" if self._path.parent and self._path.parent.name else self._path.name
            try:
                logger.info("file_in start: path=%s size=%d", str(self._path), int(self._path.stat().st_size))
            except Exception:
                logger.info("file_in start: path=%s", str(self._path))
            self._put("status", {"text": f"读取音频文件：{show_name} ..."})
            pcm16 = _load_audio_pcm16_16k_mono(self._path)
            dur_s = 0.0
            try:
                dur_s = len(pcm16) / 2.0 / 16000.0
                logger.info("file_in decoded: path=%s pcm_bytes=%d duration=%.3fs", str(self._path), len(pcm16), dur_s)
            except Exception:
                pass
            if self._stop_event.is_set():
                return

            # 文件输入：按用户需求不做 VAD 自行切分，整段音频直接做一次 ASR -> 翻译
            segments = [
                SpeechSegment(
                    segment_id=1,
                    start_time=0.0,
                    end_time=float(dur_s),
                    sample_rate=16000,
                    pcm16_bytes=pcm16,
                )
            ]

            if self._asr is None:
                self._asr, _, _, _ = build_asr(self._cfg, project_root=self._project_root)
            if self._hymt is None:
                translate_cfg = self._cfg.get("translate") if isinstance(self._cfg.get("translate"), dict) else {}
                self._hymt = HyMtClient(HyMtConfig(api_url=str(translate_cfg.get("api_url", "http://localhost:1234/v1/chat/completions"))))

            locked_code: str | None = None
            any_segment = False
            for seg in segments:
                if self._stop_event.is_set():
                    return
                asr_out = asr_segment(
                    pcm16_bytes=seg.pcm16_bytes,
                    sample_rate=seg.sample_rate,
                    asr=self._asr,  # type: ignore[arg-type]
                    direction="to_zh",
                    locked_lang_code=locked_code,
                    lock_mode="auto",
                    lock_prob_threshold=self._prob_thr,
                )
                if not asr_out:
                    continue
                text_in = str(asr_out.get("text_in", "") or "").strip()
                if text_in == "[无有效内容]":
                    continue
                asr_out["text_in"] = text_in
                any_segment = True
                if not locked_code and asr_out.get("suggested_lock_code") in LOCKABLE_LANG_CODES:
                    locked_code = str(asr_out["suggested_lock_code"]).strip().lower()
                    self._put("lock_lang", {"code": locked_code})
                self._put(
                    "asr",
                    {
                        "mode": "file_in",
                        "seg_id": int(getattr(seg, "segment_id", 0) or 0),
                        "seg_start_s": float(getattr(seg, "start_time", 0.0) or 0.0),
                        "seg_end_s": float(getattr(seg, "end_time", 0.0) or 0.0),
                        "lang_in": asr_out.get("lang_in", ""),
                        "prob": float(asr_out.get("prob", 0.0) or 0.0),
                        "text_in": text_in,
                        "t_asr_ms": float(asr_out.get("t_asr_ms", 0.0) or 0.0),
                    },
                )

                tr_out = translate_segment_text(
                    text_in=text_in,
                    lang_in=str(asr_out.get("lang_in", "") or ""),
                    hymt=self._hymt,  # type: ignore[arg-type]
                    direction="to_zh",
                    locked_lang_code=locked_code,
                )
                self._put(
                    "tr",
                    {
                        "mode": "file_in",
                        "seg_id": int(getattr(seg, "segment_id", 0) or 0),
                        "seg_start_s": float(getattr(seg, "start_time", 0.0) or 0.0),
                        "seg_end_s": float(getattr(seg, "end_time", 0.0) or 0.0),
                        "dst_code": tr_out.get("dst_code", "zh"),
                        "text_out": tr_out.get("text_out", ""),
                        "t_tr_ms": float(tr_out.get("t_tr_ms", 0.0) or 0.0),
                    },
                )

            if not any_segment:
                self._put("error", {"text": "未识别到有效语音（ASR 输出为空）。"})
            self._put("status", {"text": "文件处理完成。"})
            self._put("file_done", {"path": str(self._path), "ok": True})
        except Exception as exc:  # noqa: BLE001
            self._put("error", {"text": f"文件处理失败：{exc}"})
            self._put("file_done", {"path": str(self._path), "ok": False, "error": str(exc)})

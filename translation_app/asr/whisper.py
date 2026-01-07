"""
faster-whisper ASR 封装（GPU 优先，失败自动降级 CPU）。

输出字段：
- textA: 识别文本
- langA: 识别语言码（Whisper language code）
- lang_prob: 语言概率（0~1）
"""

from __future__ import annotations

import logging
import os
import platform
import ctypes
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsrResult:
    text: str
    language: str
    language_probability: float


class WhisperAsr:
    """faster-whisper 的轻量封装，单线程调用即可。"""

    def __init__(
        self,
        *,
        model: str = "small",
        device: str = "cuda",
        cuda_bin_dir: str | None = None,
        compute_type: str | None = None,
        cuda_suspicious_min_chars_per_s: float | None = None,
    ) -> None:
        self.model_name = self._normalize_model(model)
        self.device_preference = device
        self.cuda_bin_dir = (cuda_bin_dir or "").strip().strip('"') or None
        ct = (compute_type or "").strip().lower()
        self.compute_type_preference = (None if (not ct or ct == "auto") else ct)
        # CUDA 输出异常兜底（可配置）：当 GPU 结果“字数/时长”偏低时，自动回退 CPU 重试。
        # 设为 <=0 可关闭该判断。
        self.cuda_suspicious_min_chars_per_s = float(cuda_suspicious_min_chars_per_s if cuda_suspicious_min_chars_per_s is not None else 3.0)
        self.device_in_use: str | None = None
        self.compute_type_in_use: str | None = None
        self._model = None
        self._cpu_lang_detector = None
        self._load_model()

    @staticmethod
    def _normalize_model(model: str) -> str:
        """
        规范化模型参数：
        - 允许传入 faster-whisper 的内置模型名（如 small/large-v3 等）
        - 允许传入本地 CTranslate2 模型目录
        - 兼容用户传入 model.bin 文件路径：自动取其父目录
        """

        model = str(model).strip()
        if not model:
            return "small"

        # 若是显式路径（存在或像路径），做本地检查与归一化
        looks_like_path = ("/" in model) or ("\\" in model) or model.lower().endswith(".bin")
        if looks_like_path or os.path.exists(model):
            if os.path.isdir(model):
                return model
            if os.path.isfile(model):
                if os.path.basename(model).lower() == "model.bin":
                    return os.path.dirname(model)
                # 其它文件路径：仍返回父目录，避免直接传文件导致加载失败
                return os.path.dirname(model)
            # 看起来像路径但不存在：直接报错更可读（离线场景避免默默走下载）
            raise RuntimeError(f"本地 Whisper 模型路径不存在：{model}")

        return model

    def _check_windows_cudnn(self) -> tuple[bool, str | None]:
        """
        在 Windows 上预检查 cuDNN DLL 是否可加载。

        说明：
        - faster-whisper/ctranslate2 的 CUDA 路径在缺少 cuDNN 时可能触发底层崩溃（无法被 Python try/except 捕获）。
        - 因此在初始化阶段做 DLL 预检查，缺失则直接走 CPU，保证稳定性。
        """

        if os.name != "nt":
            return True, None

        # 优先使用 config 指定的 CUDA bin 目录（用于加载 cublas/cudart 等）
        if self.cuda_bin_dir:
            try:
                os.add_dll_directory(self.cuda_bin_dir)
                os.environ["PATH"] = f"{self.cuda_bin_dir};{os.environ.get('PATH', '')}"
                logger.info("已添加 cuda_bin_dir 到 DLL 搜索路径：%s", self.cuda_bin_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning("添加 cuda_bin_dir 失败（将继续尝试系统 PATH）：%s", exc)

        # 允许用户通过环境变量显式指定 cuDNN DLL 目录（便于离线/便携部署）
        # 示例：setx CUDNN_DLL_DIR "C:\path\to\cudnn\bin"
        cudnn_dir = (os.getenv("CUDNN_DLL_DIR") or "").strip().strip('"')
        if cudnn_dir:
            try:
                os.add_dll_directory(cudnn_dir)
                # 同时补充到 PATH，兼容依赖链的二次加载
                os.environ["PATH"] = f"{cudnn_dir};{os.environ.get('PATH', '')}"
                logger.info("已添加 CUDNN_DLL_DIR 到 DLL 搜索路径：%s", cudnn_dir)
            except Exception as exc:  # noqa: BLE001
                logger.warning("添加 CUDNN_DLL_DIR 失败（将继续尝试系统 PATH）：%s", exc)

        # 项目内置：优先把 `res/` 加入 DLL 搜索路径（如 `res/cudnn_ops64_9.dll`）
        try:
            # translation_app/asr/whisper.py -> repo/
            project_root = Path(__file__).resolve().parents[2]
            res_dir = (project_root / "res").resolve()
            if res_dir.is_dir():
                os.add_dll_directory(str(res_dir))
                os.environ["PATH"] = f"{res_dir};{os.environ.get('PATH', '')}"
        except Exception:
            pass

        # Windows GPU 路径依赖的常见 DLL 名：
        # - cuDNN 9：至少包含 ops
        # - CUDA 12：cublas 在推理阶段常会被动态加载
        required = [
            "cudnn_ops64_9.dll",
            "cublas64_12.dll",
        ]

        missing: list[str] = []
        for dll in required:
            # 先尝试加载项目 res 目录中的绝对路径（更稳定）
            try:
                res_path = None
                try:
                    project_root = Path(__file__).resolve().parents[2]
                    res_path = (project_root / "res" / dll).resolve()
                except Exception:
                    res_path = None
                if res_path is not None and res_path.is_file():
                    ctypes.WinDLL(str(res_path))
                    continue
            except OSError:
                pass
            try:
                ctypes.WinDLL(dll)
            except OSError:
                missing.append(dll)

        if missing:
            msg = (
                f"检测到缺少 GPU 依赖 DLL：{', '.join(missing)}。"
                "将自动降级为 CPU 推理。"
                "如需使用 GPU，请安装与当前 CUDA/驱动匹配的 CUDA Runtime/cuDNN，并将其 bin 目录加入 PATH。"
            )
            return False, msg

        return True, None

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel  # 延迟导入，便于错误提示
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "导入 faster-whisper 失败。请先安装：pip install faster-whisper"
            ) from exc

        if self.device_preference.lower() == "cuda":
            ok, reason = self._check_windows_cudnn()
            if not ok and reason:
                logger.warning(reason)
            if not ok:
                # Windows 缺少 cuDNN 时直接跳过 CUDA 初始化，避免后续不可捕获的底层崩溃
                self.device_preference = "cpu"
            else:
                logger.info(
                    "检测到 CUDA 预检通过：platform=%s python=%s",
                    platform.platform(),
                    platform.python_version(),
                )

        if self.device_preference.lower() == "cuda":
            # 默认优先 int8_float32：在部分 Windows + GTX 16 系列环境下更稳定（避免 float16 乱码/短句）。
            preferred = self.compute_type_preference or "int8_float32"
            candidates = [preferred]
            for c in ("int8_float32", "int8_float16", "int8", "float16"):
                if c not in candidates:
                    candidates.append(c)

            last_exc: Exception | None = None
            for ct in candidates:
                try:
                    self._model = WhisperModel(
                        self.model_name,
                        device="cuda",
                        compute_type=ct,
                    )
                    self.device_in_use = "cuda"
                    self.compute_type_in_use = ct
                    logger.info("Whisper 模型已加载：model=%s device=cuda compute_type=%s", self.model_name, ct)
                    return
                except Exception as exc:  # noqa: BLE001 - GPU 失败必须可恢复
                    last_exc = exc
                    logger.warning("Whisper GPU 初始化失败（compute_type=%s），将尝试回退：%s", ct, exc)

            logger.warning("Whisper GPU 初始化全部失败，将自动降级 CPU：%s", last_exc)

        # CPU 路径（无论用户指定 cpu，还是 cuda 失败）
        try:
            self._model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type="int8",
            )
            self.device_in_use = "cpu"
            self.compute_type_in_use = "int8"
            logger.info("Whisper 模型已加载：model=%s device=cpu compute_type=int8", self.model_name)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Whisper 模型加载失败。若你处于离线环境，请确保模型已预先下载到本机缓存或提供本地模型路径。"
            ) from exc

    def _ensure_cpu_lang_detector(self):
        if self._cpu_lang_detector is not None:
            return self._cpu_lang_detector
        from faster_whisper import WhisperModel  # noqa: WPS433

        self._cpu_lang_detector = WhisperModel(
            self.model_name,
            device="cpu",
            compute_type="int8",
        )
        return self._cpu_lang_detector

    def transcribe(self, *, pcm16_bytes: bytes, sample_rate: int = 16000, language: str | None = None) -> AsrResult:
        """对单个语音段做转写与语种识别。"""

        if not pcm16_bytes:
            return AsrResult(text="", language="", language_probability=0.0)
        if sample_rate != 16000:
            raise ValueError("当前实现要求 sample_rate=16000Hz。")

        audio = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        duration_s = float(audio.size) / 16000.0 if audio.size else 0.0

        def _gpu_output_suspicious(text: str) -> bool:
            t = (text or "").strip()
            if not t:
                return False
            thr = float(self.cuda_suspicious_min_chars_per_s)
            if thr > 0.0 and duration_s >= 1.0:
                if (len(t) / float(duration_s)) < thr:
                    return True
            # 兜底：大量重复的“乱码/符号”通常是 GPU 异常输出
            if len(t) >= 60:
                from collections import Counter

                c, n = Counter(t).most_common(1)[0]
                if (n / float(len(t))) >= 0.6 and (not c.isalnum()) and (not c.isspace()):
                    return True
            return False

        def _do_transcribe_on(model, lang: str | None) -> AsrResult:  # noqa: ANN001
            segments, info = model.transcribe(
                audio,
                task="transcribe",
                language=lang,
                vad_filter=False,
                condition_on_previous_text=False,
                beam_size=5,
            )
            text = "".join(seg.text for seg in segments).strip()
            lang = getattr(info, "language", "") or ""
            lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
            return AsrResult(text=text, language=lang, language_probability=lang_prob)

        # 极短音频段：直接丢弃（返回空），避免 Windows+CUDA 下潜在的 native 崩溃。
        if 0.0 < duration_s < 1.0:
            return AsrResult(text="", language="", language_probability=0.0)

        def _do_transcribe(lang: str | None) -> AsrResult:
            return _do_transcribe_on(self._model, lang)  # type: ignore[arg-type]

        def _transcribe_with_cuda_fallback(*, lang_for_transcribe: str | None, force_lang: str | None = None, force_prob: float | None = None) -> AsrResult:
            try:
                res = _do_transcribe(lang_for_transcribe)
            except Exception as exc:  # noqa: BLE001
                if (self.device_in_use or "").lower() != "cuda":
                    raise
                logger.warning("Whisper CUDA 推理异常，将回退 CPU 重试：%s", exc)
                detector = self._ensure_cpu_lang_detector()
                res = _do_transcribe_on(detector, lang_for_transcribe)

            # 经验兜底：某些 CUDA + compute_type 组合可能出现“转写结果为空但音频非空”的情况。
            # 此时自动回退 CPU 重试，避免上层表现为“没输出”。
            if (self.device_in_use or "").lower() == "cuda" and (not (res.text or "").strip()) and duration_s >= 1.0:
                try:
                    logger.warning(
                        "Whisper CUDA 输出为空（ct=%s dur=%.2fs），将降级 CPU 重试。建议在 config.json 设置 asr.compute_type=int8_float32。",
                        (self.compute_type_in_use or ""),
                        duration_s,
                    )
                except Exception:
                    pass
                detector = self._ensure_cpu_lang_detector()
                res = _do_transcribe_on(detector, lang_for_transcribe if lang_for_transcribe else None)

            if (self.device_in_use or "").lower() != "cuda":
                return AsrResult(
                    text=res.text,
                    language=(force_lang if force_lang is not None else res.language),
                    language_probability=(float(force_prob) if force_prob is not None else res.language_probability),
                )

            if not _gpu_output_suspicious(res.text):
                return AsrResult(
                    text=res.text,
                    language=(force_lang if force_lang is not None else res.language),
                    language_probability=(float(force_prob) if force_prob is not None else res.language_probability),
                )

            logger.warning(
                "Whisper CUDA 输出疑似异常（ct=%s dur=%.2fs len=%d），将降级 CPU 重试。建议在 config.json 设置 asr.compute_type=int8_float32。",
                (self.compute_type_in_use or ""),
                duration_s,
                len((res.text or "").strip()),
            )
            detector = self._ensure_cpu_lang_detector()
            res3 = _do_transcribe_on(detector, lang_for_transcribe if lang_for_transcribe else None)
            return AsrResult(
                text=res3.text,
                language=(force_lang if force_lang is not None else res3.language),
                language_probability=(float(force_prob) if force_prob is not None else res3.language_probability),
            )

        if language:
            return _transcribe_with_cuda_fallback(lang_for_transcribe=language)

        # 语种检测策略：
        # - 配置 device=cuda 且实际使用 CUDA：优先 GPU detect_language；若异常则回退 CPU detect_language。
        # - 配置 device=cpu 或实际使用 CPU：使用 CPU detect_language；若异常则回退到 transcribe 自动识别。
        if (self.device_in_use or "").lower() == "cuda":
            try:
                det_lang, det_prob, _ = self._model.detect_language(audio, vad_filter=False)  # type: ignore[union-attr]
                det_lang = (det_lang or "").strip().lower()
                det_prob = float(det_prob or 0.0)
                logger.info("Whisper CUDA lang detect: lang=%s prob=%.2f", det_lang, det_prob)
                if det_lang:
                    return _transcribe_with_cuda_fallback(
                        lang_for_transcribe=det_lang,
                        force_lang=det_lang,
                        force_prob=det_prob,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Whisper CUDA 语种识别失败，将回退 CPU 语种识别：%s", exc)
                try:
                    detector = self._ensure_cpu_lang_detector()
                    det_lang, det_prob, _ = detector.detect_language(audio, vad_filter=False)
                    det_lang = (det_lang or "").strip().lower()
                    det_prob = float(det_prob or 0.0)
                    logger.info("Whisper CPU lang detect: lang=%s prob=%.2f", det_lang, det_prob)
                    if det_lang:
                        return _transcribe_with_cuda_fallback(
                            lang_for_transcribe=det_lang,
                            force_lang=det_lang,
                            force_prob=det_prob,
                        )
                except Exception as exc2:  # noqa: BLE001
                    logger.warning("Whisper CPU 语种识别失败，将回退 transcribe 自动识别：%s", exc2)
        else:
            try:
                det_lang, det_prob, _ = self._model.detect_language(audio, vad_filter=False)  # type: ignore[union-attr]
                det_lang = (det_lang or "").strip().lower()
                det_prob = float(det_prob or 0.0)
                logger.info("Whisper CPU lang detect: lang=%s prob=%.2f", det_lang, det_prob)
                if det_lang:
                    return _transcribe_with_cuda_fallback(
                        lang_for_transcribe=det_lang,
                        force_lang=det_lang,
                        force_prob=det_prob,
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Whisper CPU 语种识别失败，将回退 transcribe 自动识别：%s", exc)

        return _transcribe_with_cuda_fallback(lang_for_transcribe=None)


def _whisper_asr_worker(  # pragma: no cover
    conn,  # multiprocessing.connection.Connection (runtime)
    *,
    model: str,
    device: str,
    cuda_bin_dir: str | None,
    compute_type: str | None,
    cuda_suspicious_min_chars_per_s: float | None,
) -> None:
    """
    独立子进程 ASR Worker：用于隔离 Windows+CUDA 下可能发生的 native 崩溃（0xC0000409），避免拖垮主进程。
    """

    try:
        asr = WhisperAsr(
            model=model,
            device=device,
            cuda_bin_dir=cuda_bin_dir,
            compute_type=compute_type,
            cuda_suspicious_min_chars_per_s=cuda_suspicious_min_chars_per_s,
        )
    except Exception as exc:  # noqa: BLE001
        try:
            conn.send({"ok": False, "error": f"init_failed: {exc}"})
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        return

    try:
        conn.send({"ok": True})
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return

    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        except Exception:
            break
        if not isinstance(msg, dict):
            continue
        mtype = str(msg.get("type", "") or "")
        if mtype == "close":
            break
        if mtype != "transcribe":
            continue
        try:
            pcm = bytes(msg.get("pcm16_bytes") or b"")
            sr = int(msg.get("sample_rate") or 16000)
            lang = msg.get("language")
            lang = (str(lang).strip().lower() if lang else None)
            res = asr.transcribe(pcm16_bytes=pcm, sample_rate=sr, language=lang)
            conn.send({"ok": True, "text": res.text, "language": res.language, "language_probability": float(res.language_probability)})
        except Exception as exc:  # noqa: BLE001
            try:
                conn.send({"ok": False, "error": str(exc)})
            except Exception:
                pass

    try:
        conn.close()
    except Exception:
        pass


class IsolatedWhisperAsr:
    """
    进程隔离的 Whisper ASR：
    - 子进程负责加载/运行 faster-whisper（可用 CUDA）
    - 若子进程发生 native crash，主进程可感知并自动回退 CPU，避免整个服务退出
    """

    def __init__(
        self,
        *,
        model: str = "small",
        device: str = "cuda",
        cuda_bin_dir: str | None = None,
        compute_type: str | None = None,
        cuda_suspicious_min_chars_per_s: float | None = None,
    ) -> None:
        self.model_name = str(model)
        self.device_preference = str(device)
        self.cuda_bin_dir = (cuda_bin_dir or "").strip().strip('"') or None
        self.compute_type_preference = (str(compute_type).strip().lower() if compute_type else None)
        self.cuda_suspicious_min_chars_per_s = cuda_suspicious_min_chars_per_s
        self._cpu_fallback: WhisperAsr | None = None
        self._ctx = None
        self._conn = None
        self._proc = None
        self._start_worker()

    def _start_worker(self) -> None:
        try:
            import multiprocessing as mp

            self._ctx = mp.get_context("spawn")
            parent_conn, child_conn = self._ctx.Pipe(duplex=True)
            proc = self._ctx.Process(
                target=_whisper_asr_worker,
                kwargs={
                    "conn": child_conn,
                    "model": self.model_name,
                    "device": self.device_preference,
                    "cuda_bin_dir": self.cuda_bin_dir,
                    "compute_type": self.compute_type_preference,
                    "cuda_suspicious_min_chars_per_s": self.cuda_suspicious_min_chars_per_s,
                },
                daemon=True,
                name="WhisperAsrWorker",
            )
            proc.start()
            child_conn.close()

            # Wait for init ack (or worker crash) to avoid hanging on first call.
            # If CUDA init crashes the worker, recv() would block forever without a poll timeout.
            deadline = time.time() + 120.0
            ack = None
            while True:
                if parent_conn.poll(0.2):
                    ack = parent_conn.recv()
                    break
                if not proc.is_alive():
                    break
                if time.time() >= deadline:
                    break
            if ack is None:
                raise RuntimeError("worker init timeout/crash")
            if not (isinstance(ack, dict) and ack.get("ok") is True):
                err = ""
                try:
                    err = str((ack or {}).get("error", "") or "")
                except Exception:
                    err = ""
                raise RuntimeError(err or "worker init failed")

            self._proc = proc
            self._conn = parent_conn
        except Exception as exc:  # noqa: BLE001
            try:
                logger.warning("IsolatedWhisperAsr init failed, fallback to CPU: %s", exc)
            except Exception:
                pass
            self._cpu_fallback = WhisperAsr(model=self.model_name, device="cpu", cuda_bin_dir=None, compute_type="int8")

    def _ensure_worker(self) -> bool:
        if self._cpu_fallback is not None:
            return False
        if self._proc is None or self._conn is None:
            return False
        try:
            if not self._proc.is_alive():
                return False
        except Exception:
            return False
        return True

    def _fallback_cpu(self) -> WhisperAsr:
        if self._cpu_fallback is None:
            self._cpu_fallback = WhisperAsr(model=self.model_name, device="cpu", cuda_bin_dir=None, compute_type="int8")
        return self._cpu_fallback

    def transcribe(self, *, pcm16_bytes: bytes, sample_rate: int = 16000, language: str | None = None) -> AsrResult:
        if not pcm16_bytes:
            return AsrResult(text="", language="", language_probability=0.0)
        if sample_rate != 16000:
            raise ValueError("当前实现要求 sample_rate=16000Hz。")

        # 与上层保持一致：极短音频直接丢弃。
        try:
            duration_s = float(len(pcm16_bytes)) / (2.0 * 16000.0)
        except Exception:
            duration_s = 0.0
        if 0.0 < duration_s < 1.0:
            return AsrResult(text="", language="", language_probability=0.0)

        if self._cpu_fallback is not None:
            return self._cpu_fallback.transcribe(pcm16_bytes=pcm16_bytes, sample_rate=sample_rate, language=language)

        if not self._ensure_worker():
            try:
                logger.warning("Whisper worker not alive; fallback to CPU.")
            except Exception:
                pass
            return self._fallback_cpu().transcribe(pcm16_bytes=pcm16_bytes, sample_rate=sample_rate, language=language)

        try:
            self._conn.send({"type": "transcribe", "pcm16_bytes": pcm16_bytes, "sample_rate": int(sample_rate), "language": language})
            resp = self._conn.recv()
        except Exception:
            try:
                logger.warning("Whisper worker IPC failed; fallback to CPU.")
            except Exception:
                pass
            return self._fallback_cpu().transcribe(pcm16_bytes=pcm16_bytes, sample_rate=sample_rate, language=language)

        if not (isinstance(resp, dict) and resp.get("ok") is True):
            try:
                logger.warning("Whisper worker returned error; fallback to CPU: %s", (resp or {}).get("error", ""))
            except Exception:
                pass
            return self._fallback_cpu().transcribe(pcm16_bytes=pcm16_bytes, sample_rate=sample_rate, language=language)

        return AsrResult(
            text=str(resp.get("text", "") or ""),
            language=str(resp.get("language", "") or ""),
            language_probability=float(resp.get("language_probability", 0.0) or 0.0),
        )

    def close(self) -> None:
        try:
            if self._conn is not None:
                try:
                    self._conn.send({"type": "close"})
                except Exception:
                    pass
        finally:
            try:
                if self._conn is not None:
                    self._conn.close()
            except Exception:
                pass
            self._conn = None
            try:
                if self._proc is not None:
                    self._proc.join(timeout=0.5)
            except Exception:
                pass
            self._proc = None

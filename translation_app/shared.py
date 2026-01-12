"""
共享逻辑层：供 Web 服务复用（`translation_app.shared`）。

目标：
- 把“配置读取/保存、ASR/翻译/TTS 初始化、语种锁定选项”等稳定逻辑集中管理
"""

from __future__ import annotations

import logging
import json
import os
import time
from pathlib import Path

from translation_app.translate.hymt import HyMtClient, HyMtConfig, LANG_CODE_TO_NAME
from translation_app.translate.ollama_correct import AsrCorrector, AsrCorrectorConfig, AsrCorrectorError
from translation_app.tts.engine import TtsConfig, TtsEngine

logger = logging.getLogger(__name__)

# 前端语种下拉：这些视为“可用/稳定”（置顶并加粗），其它视为实验状态。
STABLE_LANG_CODES: set[str] = {"en", "es", "ar", "fr", "ru", "ja", "ko", "de", "pt", "vi", "th", "id", "ur", "fa"}

LOCK_LANG_CHOICES: list[tuple[str, str]] = [
    ("en", "英文"),
    ("es", "西班牙语"),
    ("ar", "阿拉伯语"),
    ("fr", "法语"),
    ("ru", "俄语"),
    ("ja", "日语"),
    ("ko", "韩语"),
    ("de", "德语"),
    ("pt", "葡萄牙语"),
    ("yue", "粤语"),
    ("it", "意大利语"),
    ("tr", "土耳其语"),
    ("th", "泰语"),
    ("vi", "越南语"),
    ("ms", "马来语"),
    ("id", "印尼语"),
    ("tl", "菲律宾语/他加禄语"),
    ("hi", "印地语"),
    ("pl", "波兰语"),
    ("cs", "捷克语"),
    ("nl", "荷兰语"),
    ("km", "高棉语"),
    ("my", "缅甸语"),
    ("fa", "波斯语"),
    ("gu", "古吉拉特语"),
    ("ur", "乌尔都语"),
    ("te", "泰卢固语"),
    ("mr", "马拉地语"),
    ("he", "希伯来语"),
    ("bn", "孟加拉语"),
    ("ta", "泰米尔语"),
    ("uk", "乌克兰语"),
    ("bo", "藏语"),
    ("kk", "哈萨克语"),
    ("mn", "蒙古语"),
    ("ug", "维吾尔语"),
]

LOCK_LANG_CODE_TO_LABEL: dict[str, str] = {c: f"{n}（{c}）" for c, n in LOCK_LANG_CHOICES}
LOCKABLE_LANG_CODES: set[str] = set(LOCK_LANG_CODE_TO_LABEL.keys())


def cfg_get(d: dict, key: str, default):
    v = d.get(key, default)
    return default if v is None else v


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"缺少配置文件：{path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"读取配置文件失败：{path}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"配置文件格式错误（根节点必须是 JSON 对象）：{path}")
    return data


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def resolve_rel_path(*, project_root: Path, rel_or_abs: str) -> Path:
    p = Path(str(rel_or_abs).strip())
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def relpath_under_root(*, project_root: Path, abs_path: Path) -> str | None:
    try:
        rel = abs_path.resolve().relative_to(project_root.resolve())
    except Exception:
        return None
    return rel.as_posix()


def build_hymt(cfg: dict) -> HyMtClient:
    translate_cfg = cfg_get(cfg, "translate", {}) if isinstance(cfg_get(cfg, "translate", {}), dict) else {}
    api_url = str(cfg_get(translate_cfg, "api_url", "http://localhost:1234/v1/chat/completions")).strip()
    return HyMtClient(HyMtConfig(api_url=api_url))


def build_asr_corrector(cfg: dict) -> AsrCorrector | None:
    corr_cfg = cfg_get(cfg, "asr_correction", {}) if isinstance(cfg_get(cfg, "asr_correction", {}), dict) else {}
    enabled = bool(cfg_get(corr_cfg, "enabled", True))
    if not enabled:
        return None
    api_url = str(cfg_get(corr_cfg, "api_url", "http://localhost:11434/v1/chat/completions")).strip()
    model = str(cfg_get(corr_cfg, "model", "gemma3:4b")).strip()
    temperature = float(cfg_get(corr_cfg, "temperature", 0.2) or 0.2)
    connect_timeout_s = float(cfg_get(corr_cfg, "connect_timeout_s", 3.0) or 3.0)
    read_timeout_s = float(cfg_get(corr_cfg, "read_timeout_s", 60.0) or 60.0)
    max_retries = int(cfg_get(corr_cfg, "max_retries", 1) or 1)
    backoff_s = float(cfg_get(corr_cfg, "backoff_s", 0.4) or 0.4)
    api_key = str(cfg_get(corr_cfg, "api_key", "") or "").strip() or None
    return AsrCorrector(
        AsrCorrectorConfig(
            api_url=api_url,
            model=model,
            temperature=temperature,
            connect_timeout_s=connect_timeout_s,
            read_timeout_s=read_timeout_s,
            max_retries=max_retries,
            backoff_s=backoff_s,
            api_key=api_key,
        )
    )


def correct_asr_text(*, text_in: str, corrector: AsrCorrector | None) -> str | None:
    text = str(text_in or "").strip()
    if not text:
        return None
    if corrector is None:
        return text
    try:
        corrected = corrector.correct(text)
    except AsrCorrectorError as exc:
        logger.warning("ASR 语义纠错失败，使用原文：%s", exc)
        return text
    if corrected is None:
        return None
    return corrected


def build_tts(cfg: dict) -> TtsEngine:
    tts_cfg = cfg_get(cfg, "tts", {}) if isinstance(cfg_get(cfg, "tts", {}), dict) else {}
    enabled = bool(cfg_get(tts_cfg, "enabled", True))
    backend = str(cfg_get(tts_cfg, "backend", "auto")).strip().lower()
    return TtsEngine(TtsConfig(backend=("disabled" if not enabled else backend)))


def build_asr(cfg: dict, *, project_root: Path):
    asr_cfg = cfg_get(cfg, "asr", {}) if isinstance(cfg_get(cfg, "asr", {}), dict) else {}
    backend = str(cfg_get(asr_cfg, "backend", "whisper")).strip().lower()
    device = str(cfg_get(asr_cfg, "device", "cuda")).strip().lower()
    model_path = str(cfg_get(asr_cfg, "model_path", "")).strip()
    cuda_bin_dir = str(cfg_get(asr_cfg, "cuda_bin_dir", "")).strip()
    compute_type = str(cfg_get(asr_cfg, "compute_type", "")).strip()
    isolate = cfg_get(asr_cfg, "isolate_process", None)
    suspicious_min_chars_per_s = cfg_get(asr_cfg, "cuda_suspicious_min_chars_per_s", None)
    if backend != "whisper":
        raise RuntimeError(f"配置错误：asr.backend 仅支持 whisper（当前={backend}）")
    if device not in ("cuda", "cpu"):
        raise RuntimeError(f"配置错误：asr.device 必须是 cuda 或 cpu（当前={device}）")

    # model_path 可选：
    # - 空：使用 faster-whisper 内置模型名 "small"
    # - 路径（含 /、\ 或以 .bin 结尾）：按项目根目录解析并要求存在
    # - 其它：按 faster-whisper 内置模型名处理（如 small/large-v3 等）
    model_id = model_path or "small"
    looks_like_path = ("/" in model_id) or ("\\" in model_id) or model_id.lower().endswith(".bin")
    model_dir: Path | None = None
    if looks_like_path:
        model_dir = resolve_rel_path(project_root=project_root, rel_or_abs=model_id)
        if not model_dir.exists():
            raise RuntimeError(f"ASR 模型路径不存在：{model_dir}")
        model_id = str(model_dir)
    else:
        # 兼容用户填写项目内目录名（如 faster-whisper-large-v3-turbo / faster-whisper-small）
        candidate = resolve_rel_path(project_root=project_root, rel_or_abs=model_id)
        if candidate.exists():
            model_dir = candidate
            model_id = str(candidate)

    from translation_app.asr.whisper import IsolatedWhisperAsr, WhisperAsr  # noqa: WPS433

    if isolate is None:
        # Windows + CUDA：默认启用进程隔离，避免 faster-whisper/ctranslate2 的 native 崩溃直接导致主进程退出。
        isolate = (os.name == "nt" and device == "cuda")
    isolate = bool(isolate and device == "cuda")

    asr = (
        IsolatedWhisperAsr(
            model=model_id,
            device=device,
            cuda_bin_dir=(cuda_bin_dir or None),
            compute_type=(compute_type or None),
            cuda_suspicious_min_chars_per_s=(
                float(suspicious_min_chars_per_s)
                if suspicious_min_chars_per_s is not None and str(suspicious_min_chars_per_s).strip()
                else None
            ),
        )
        if isolate
        else WhisperAsr(
            model=model_id,
            device=device,
            cuda_bin_dir=(cuda_bin_dir or None),
            compute_type=(compute_type or None),
            cuda_suspicious_min_chars_per_s=(
                float(suspicious_min_chars_per_s)
                if suspicious_min_chars_per_s is not None and str(suspicious_min_chars_per_s).strip()
                else None
            ),
        )
    )

    return (
        asr,
        backend,
        device,
        (model_dir or Path(model_id)),
    )


def pretty_lang_label(code: str | None) -> str:
    if not code:
        return ""
    code = code.strip().lower()
    return LOCK_LANG_CODE_TO_LABEL.get(code) or LANG_CODE_TO_NAME.get(code) or code


class AsrProtocol:  # minimal runtime protocol
    def transcribe(self, *, pcm16_bytes: bytes, sample_rate: int = 16000):  # noqa: ANN201
        raise NotImplementedError


def asr_segment(
    *,
    pcm16_bytes: bytes,
    sample_rate: int,
    asr: AsrProtocol,
    direction: str,  # "to_zh" | "to_locked"
    locked_lang_code: str | None = None,
    lock_mode: str = "auto",  # "auto" | "manual"
    lock_prob_threshold: float = 0.7,
) -> dict | None:
    if not pcm16_bytes:
        return None
    if int(sample_rate) != 16000:
        raise ValueError("当前实现要求 sample_rate=16000Hz。")
    # 丢弃过短音频段：短段通常来自 VAD 误触发，且在部分 Windows+CUDA 环境可能触发底层崩溃。
    # 正常使用场景下不应频繁出现 <1s 的有效语音段。
    dur_s = float(len(pcm16_bytes)) / (2.0 * float(sample_rate)) if pcm16_bytes else 0.0
    if 0.0 < dur_s < 1.0:
        logger.info("drop short segment: dur=%.3fs", dur_s)
        return None

    asr_lang_hint: str | None = None
    if direction == "to_zh":
        # mic_in / file_in：锁定语种 L 表示“输入语种”，可作为 ASR 提示以跳过自动识别
        if (locked_lang_code or "").strip():
            asr_lang_hint = (locked_lang_code or "").strip().lower()
    elif direction == "to_locked":
        # mic_out：UI 语义为“中文 -> 语种 L”，因此 ASR 输入默认按中文提示
        asr_lang_hint = "zh"
    else:
        raise ValueError(f"unknown direction={direction!r}")

    t0 = time.perf_counter()
    if asr_lang_hint:
        try:
            asr_res = asr.transcribe(pcm16_bytes=pcm16_bytes, sample_rate=int(sample_rate), language=asr_lang_hint)
        except TypeError:
            asr_res = asr.transcribe(pcm16_bytes=pcm16_bytes, sample_rate=int(sample_rate))
    else:
        asr_res = asr.transcribe(pcm16_bytes=pcm16_bytes, sample_rate=int(sample_rate))
    t_asr_ms = (time.perf_counter() - t0) * 1000.0
    text_in = (getattr(asr_res, "text", "") or "").strip()
    lang_in = (getattr(asr_res, "language", "") or "").strip().lower()
    try:
        prob = float(getattr(asr_res, "language_probability", 0.0) or 0.0)
    except Exception:
        prob = 0.0

    if not text_in:
        logger.info("perf segment: asr=%.1fms (empty)", t_asr_ms)
        return None

    suggested_lock_code: str | None = None
    if (
        direction == "to_zh"
        and (lock_mode or "auto").strip().lower() == "auto"
        and not (locked_lang_code or "").strip()
        and lang_in in LOCKABLE_LANG_CODES
        and ((prob >= float(lock_prob_threshold)) or (prob <= 0.0))
    ):
        suggested_lock_code = lang_in

    return {
        "text_in": text_in,
        "lang_in": lang_in,
        "prob": prob,
        "suggested_lock_code": suggested_lock_code,
        "t_asr_ms": t_asr_ms,
    }


def translate_segment_text(
    *,
    text_in: str,
    lang_in: str,
    hymt: HyMtClient,
    direction: str,  # "to_zh" | "to_locked"
    locked_lang_code: str | None = None,
) -> dict:
    if direction == "to_zh":
        src_name = LANG_CODE_TO_NAME.get(lang_in) or "Auto Detect"
        t1 = time.perf_counter()
        text_out = hymt.translate(text_in, src_lang=src_name, dst_lang="Simplified Chinese")
        t_tr_ms = (time.perf_counter() - t1) * 1000.0
        return {"dst_code": "zh", "text_out": text_out, "t_tr_ms": t_tr_ms}

    if direction != "to_locked":
        raise ValueError(f"unknown direction={direction!r}")

    target_code = (locked_lang_code or "").strip().lower()
    target_name = LANG_CODE_TO_NAME.get(target_code)
    if not target_name:
        raise RuntimeError("尚未锁定语种 L。")

    src_name = LANG_CODE_TO_NAME.get(lang_in) or "Auto Detect"
    t1 = time.perf_counter()
    text_out = hymt.translate(text_in, src_lang=src_name, dst_lang=target_name)
    t_tr_ms = (time.perf_counter() - t1) * 1000.0
    return {"dst_code": target_code, "text_out": text_out, "t_tr_ms": t_tr_ms}


def process_segment(
    *,
    pcm16_bytes: bytes,
    sample_rate: int,
    asr: AsrProtocol,
    hymt: HyMtClient,
    tts: TtsEngine | None,
    direction: str,  # "to_zh" | "to_locked"
    locked_lang_code: str | None = None,
    lock_mode: str = "auto",  # "auto" | "manual"
    lock_prob_threshold: float = 0.7,
    tts_enabled: bool = False,
) -> dict | None:
    """
    统一的“音频段处理流程”：
    - ASR -> 翻译 ->（可选）锁定语种 ->（可选）TTS

    返回：
    - None：表示该段无有效文本
    - dict：包含 text_in/lang_in/prob/text_out/dst_code/tts_played/suggested_lock_code 等
    """

    t0 = time.perf_counter()
    asr_out = asr_segment(
        pcm16_bytes=pcm16_bytes,
        sample_rate=sample_rate,
        asr=asr,
        direction=direction,
        locked_lang_code=locked_lang_code,
        lock_mode=lock_mode,
        lock_prob_threshold=lock_prob_threshold,
    )
    if not asr_out:
        return None
    text_in = str(asr_out.get("text_in", "") or "").strip()
    lang_in = str(asr_out.get("lang_in", "") or "").strip().lower()
    prob = float(asr_out.get("prob", 0.0) or 0.0)
    suggested_lock_code = asr_out.get("suggested_lock_code")
    t_asr_ms = float(asr_out.get("t_asr_ms", 0.0) or 0.0)

    tr_out = translate_segment_text(
        text_in=text_in,
        lang_in=lang_in,
        hymt=hymt,
        direction=direction,
        locked_lang_code=locked_lang_code,
    )
    text_out = str(tr_out.get("text_out", "") or "")
    dst_code = str(tr_out.get("dst_code", "") or "")
    t_tr_ms = float(tr_out.get("t_tr_ms", 0.0) or 0.0)

    if direction == "to_zh":
        logger.info(
            "perf segment to_zh: asr=%.1fms tr=%.1fms total=%.1fms lang=%s prob=%.2f in_len=%d out_len=%d",
            t_asr_ms,
            t_tr_ms,
            (time.perf_counter() - t0) * 1000.0,
            (lang_in or "unk"),
            prob,
            len(text_in),
            len(text_out or ""),
        )
        return {
            "text_in": text_in,
            "lang_in": lang_in,
            "prob": prob,
            "dst_code": "zh",
            "text_out": text_out,
            "tts_played": False,
            "suggested_lock_code": suggested_lock_code,
            "t_asr_ms": t_asr_ms,
            "t_tr_ms": t_tr_ms,
        }

    played = False
    t_tts_ms = 0.0
    if tts_enabled and tts is not None and text_out:
        t2 = time.perf_counter()
        played = bool(tts.speak(text_out))
        t_tts_ms = (time.perf_counter() - t2) * 1000.0
    logger.info(
        "perf segment to_locked: asr=%.1fms tr=%.1fms tts=%.1fms total=%.1fms lang=%s->%s prob=%.2f in_len=%d out_len=%d",
        t_asr_ms,
        t_tr_ms,
        t_tts_ms,
        (time.perf_counter() - t0) * 1000.0,
        (lang_in or "unk"),
        dst_code,
        prob,
        len(text_in),
        len(text_out or ""),
    )
    return {
        "text_in": text_in,
        "lang_in": lang_in,
        "prob": prob,
        "dst_code": dst_code,
        "text_out": text_out,
        "tts_played": played,
        "suggested_lock_code": suggested_lock_code,
        "t_asr_ms": t_asr_ms,
        "t_tr_ms": t_tr_ms,
        "t_tts_ms": t_tts_ms,
    }


def process_text(
    *,
    text_zh: str,
    hymt: HyMtClient,
    tts: TtsEngine | None,
    locked_lang_code: str | None,
    tts_enabled: bool = False,
) -> dict | None:
    """
    统一的“文本输出流程”：
    - 中文文本 -> 语种 L ->（可选）TTS
    """

    text_zh = (text_zh or "").strip()
    if not text_zh:
        return None

    target_code = (locked_lang_code or "").strip().lower()
    target_name = LANG_CODE_TO_NAME.get(target_code)
    if not target_name:
        raise RuntimeError("尚未锁定语种 L。")

    t0 = time.perf_counter()
    text_out = hymt.translate(text_zh, src_lang="Simplified Chinese", dst_lang=target_name)
    t_tr_ms = (time.perf_counter() - t0) * 1000.0
    played = False
    t_tts_ms = 0.0
    if tts_enabled and tts is not None and text_out:
        t1 = time.perf_counter()
        played = bool(tts.speak(text_out))
        t_tts_ms = (time.perf_counter() - t1) * 1000.0
    logger.info(
        "perf text_out: tr=%.1fms tts=%.1fms total=%.1fms dst=%s in_len=%d out_len=%d",
        t_tr_ms,
        t_tts_ms,
        (time.perf_counter() - t0) * 1000.0,
        target_code,
        len(text_zh),
        len(text_out or ""),
    )

    return {
        "text_in": text_zh,
        "lang_in": "zh",
        "prob": 1.0,
        "dst_code": target_code,
        "text_out": text_out,
        "tts_played": played,
        "t_tr_ms": t_tr_ms,
        "t_tts_ms": t_tts_ms,
    }

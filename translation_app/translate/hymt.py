"""
HY-MT 本地翻译接口封装。

硬性约束：
- 只允许访问用户指定的本地 OpenAI 兼容接口（默认 http://localhost:1234/v1/chat/completions）。
- 必须带超时与重试，错误信息可读，便于 CLI 稳定运行。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)


LANG_CODE_TO_NAME: dict[str, str] = {
    # faster-whisper/Whisper 常见语言码 -> 目标语言名（按用户要求）
    "zh": "Simplified Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ar": "Arabic",
    # 扩展语种（用于回译目标语言 L 的手动选择）
    "yue": "Cantonese",
    "pt": "Portuguese",
    "it": "Italian",
    "ru": "Russian",
    "tr": "Turkish",
    "th": "Thai",
    "vi": "Vietnamese",
    "ms": "Malay",
    "id": "Indonesian",
    "tl": "Filipino",
    "hi": "Hindi",
    "pl": "Polish",
    "cs": "Czech",
    "nl": "Dutch",
    "km": "Khmer",
    "my": "Burmese",
    "fa": "Persian",
    "gu": "Gujarati",
    "ur": "Urdu",
    "te": "Telugu",
    "mr": "Marathi",
    "he": "Hebrew",
    "bn": "Bengali",
    "ta": "Tamil",
    "uk": "Ukrainian",
    "bo": "Tibetan",
    "kk": "Kazakh",
    "mn": "Mongolian",
    "ug": "Uyghur",
}


def lang_code_to_name(lang_code: str | None) -> str | None:
    """将 Whisper 语言码映射为 HY-MT 目标语言名；未知则返回 None。"""

    if not lang_code:
        return None
    return LANG_CODE_TO_NAME.get(lang_code.strip().lower())


@dataclass(frozen=True)
class HyMtConfig:
    api_url: str = "http://localhost:1234/v1/chat/completions"
    model: str = "hy-mt1.5-1.8b"
    temperature: float = 0.2
    max_tokens: int = -1
    stream: bool = False
    connect_timeout_s: float = 3.0
    read_timeout_s: float = 120.0
    max_retries: int = 2
    backoff_s: float = 0.6
    api_key: str | None = None


class HyMtError(RuntimeError):
    """HY-MT 调用失败（网络/协议/返回结构不符合预期）。"""


class HyMtClient:
    """
    通过 OpenAI 兼容的 chat/completions 接口调用本地 HY-MT 翻译模型。

    对外暴露：translate(text, src_lang, dst_lang) -> str
    """

    def __init__(self, config: HyMtConfig):
        self._cfg = config
        self._session = requests.Session()
        # requests.Session 并不保证线程安全；CLI 中推理线程与主线程可能并发调用翻译
        self._lock = threading.Lock()

    def translate(self, text: str, src_lang: str, dst_lang: str) -> str:
        """
        翻译文本。

        参数：
        - text: 待翻译文本
        - src_lang: 源语言名（例如 English/Japanese/... 或 Auto Detect）
        - dst_lang: 目标语言名（例如 Simplified Chinese/English/...）
        """

        if text is None:
            text = ""
        text = str(text).strip()
        if not text:
            return ""

        system_prompt = (
            "你是专业翻译，只输出译文，不要解释，不要加引号，不要换行以外的额外内容。"
        )
        user_prompt = (
            f"请将下面文本从 {src_lang} 翻译为 {dst_lang}，只输出译文：\n{text}"
        )

        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._cfg.temperature,
            "max_tokens": self._cfg.max_tokens,
            "stream": self._cfg.stream,
        }

        headers = {"Content-Type": "application/json"}
        api_key = self._cfg.api_key or os.getenv("HYMT_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        last_error: Exception | None = None
        for attempt in range(self._cfg.max_retries + 1):
            try:
                with self._lock:
                    resp = self._session.post(
                        self._cfg.api_url,
                        headers=headers,
                        json=payload,
                        timeout=(self._cfg.connect_timeout_s, self._cfg.read_timeout_s),
                    )
                if resp.status_code >= 400:
                    raise HyMtError(
                        f"HY-MT HTTP {resp.status_code}: {resp.text[:500]}"
                    )
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if content is None:
                    content = ""
                return str(content).strip()
            except (requests.Timeout, requests.ConnectionError, requests.RequestException) as exc:
                last_error = exc
                if attempt >= self._cfg.max_retries:
                    break
                sleep_s = self._cfg.backoff_s * (2**attempt)
                logger.warning(
                    "HY-MT 请求失败，将重试（第 %d/%d 次）：%s",
                    attempt + 1,
                    self._cfg.max_retries + 1,
                    exc,
                )
                time.sleep(sleep_s)
            except ValueError as exc:
                # JSON 解析失败
                raise HyMtError(f"HY-MT 返回非 JSON：{exc}") from exc
            except HyMtError:
                raise
            except Exception as exc:  # noqa: BLE001 - 需要可读兜底
                raise HyMtError(f"HY-MT 调用异常：{exc}") from exc

        raise HyMtError(
            f"HY-MT 请求失败（已重试 {self._cfg.max_retries} 次）：{last_error}"
        )

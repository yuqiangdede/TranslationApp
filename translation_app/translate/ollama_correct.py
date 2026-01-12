"""
本地语义纠错（用于 ASR 转写的纠错），使用 OpenAI 兼容接口。

硬性约束：
- 只允许访问本地 OpenAI 兼容接口（默认 http://localhost:11434/v1/chat/completions）。
- 必须带超时与重试，错误信息可读，便于 CLI 稳定运行。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

logger = logging.getLogger(__name__)

_LOG_TEXT_LIMIT = 2000


def _clip_log_text(text: str, *, limit: int = _LOG_TEXT_LIMIT) -> str:
    s = str(text or "")
    if len(s) <= limit:
        return s
    return f"{s[:limit]}...<truncated {len(s) - limit} chars>"

_CORRECT_PROMPT = (
    "你是中文语音识别结果的语义纠错器。\n"
    "输入是一段 ASR 转写文本，可能存在同音字、多音字、口语化、省略、错别字、断句错误。\n"
    "\n"
    "任务要求：\n"
    "\n"
    "在不改变原意的前提下，修正明显的同音字、多音字、常见识别错误，使语义通顺、自然。\n"
    "\n"
    "合并或调整不合理的断句，不要增加原文不存在的信息。\n"
    "\n"
    "过滤无实际语义的内容，如连续语气词、重复词、口头填充词（如“嗯”“啊”“那个”“就是”“然后然后”）。\n"
    "\n"
    "若整段文本不构成有效语义，直接输出：[无有效内容]\n"
    "\n"
    "输出要求：\n"
    "\n"
    "只输出纠正后的文本\n"
    "\n"
    "不要解释、不标注、不对照原文\n"
)


@dataclass(frozen=True)
class AsrCorrectorConfig:
    api_url: str = "http://localhost:11434/v1/chat/completions"
    model: str = "gemma3:4b"
    temperature: float = 0.2
    connect_timeout_s: float = 3.0
    read_timeout_s: float = 60.0
    max_retries: int = 1
    backoff_s: float = 0.4
    api_key: str | None = None


class AsrCorrectorError(RuntimeError):
    """ASR 语义纠错失败（网络/协议/返回结构不符合预期）。"""


class AsrCorrector:
    """通过 Ollama /api/generate 进行 ASR 语义纠错。"""

    def __init__(self, config: AsrCorrectorConfig):
        self._cfg = config
        self._session = requests.Session()
        self._lock = threading.Lock()

    def _build_prompt(self, text: str) -> str:
        return f"{_CORRECT_PROMPT}\n\n输入：\n{text}\n\n输出："

    def correct(self, text: str) -> str | None:
        if text is None:
            text = ""
        text = str(text).strip()
        if not text:
            return None
        if text == "[无有效内容]":
            return None

        system_prompt = _CORRECT_PROMPT
        user_prompt = f"{text}"
        payload: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self._cfg.temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self._cfg.api_key:
            headers["Authorization"] = f"Bearer {self._cfg.api_key}"
        logger.info(
            "ASR correction request: url=%s model=%s temperature=%s stream=%s",
            self._cfg.api_url,
            self._cfg.model,
            self._cfg.temperature,
            False,
        )
        logger.info("ASR correction prompt(system): %s", _clip_log_text(system_prompt))
        logger.info("ASR correction prompt(user): %s", _clip_log_text(user_prompt))

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
                    raise AsrCorrectorError(
                        f"Ollama HTTP {resp.status_code}: {resp.text[:500]}"
                    )
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if content is None:
                    content = ""
                content = str(content).strip()
                if not content:
                    raise AsrCorrectorError("纠错模型返回空结果。")
                logger.info("ASR correction response: %s", _clip_log_text(content))
                if content == "[无有效内容]":
                    return None
                return content
            except (requests.Timeout, requests.ConnectionError, requests.RequestException) as exc:
                last_error = exc
                if attempt >= self._cfg.max_retries:
                    break
                sleep_s = self._cfg.backoff_s * (2**attempt)
                logger.warning(
                    "ASR 语义纠错失败，将重试（第 %d/%d 次）：%s",
                    attempt + 1,
                    self._cfg.max_retries + 1,
                    exc,
                )
                time.sleep(sleep_s)
            except ValueError as exc:
                raise AsrCorrectorError(f"纠错模型返回非 JSON：{exc}") from exc
            except AsrCorrectorError:
                raise
            except Exception as exc:  # noqa: BLE001 - 需要可读兜底
                raise AsrCorrectorError(f"纠错模型调用异常：{exc}") from exc

        raise AsrCorrectorError(
            f"ASR 语义纠错请求失败（已重试 {self._cfg.max_retries} 次）：{last_error}"
        )

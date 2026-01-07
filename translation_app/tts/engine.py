"""
离线 TTS（语音合成 + 播放）。

设计目标：
- 离线运行，不依赖任何外部云服务。
- Windows 10/11 优先：尽量使用系统自带能力。
- 稳定性优先：TTS 不可用时只降级为“打印文本”，不允许崩溃主程序。

后端策略（自动选择）：
1) 若环境中安装了 pywin32，则使用 SAPI（win32com）直接朗读（性能更好）
2) 否则退化为调用 powershell.exe + System.Speech（无需额外 Python 依赖，但每次会启动子进程）
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TtsConfig:
    backend: str = "auto"  # auto / sapi / powershell / disabled
    voice_name: str | None = None  # SAPI 语音名称关键字（可选）
    rate: int | None = None  # 语速（SAPI：-10~10）
    volume: int | None = None  # 音量（SAPI：0~100）
    timeout_s: float = 60.0  # powershell 子进程最长运行时间


class TtsEngine:
    """TTS 播放封装。"""

    def __init__(self, config: TtsConfig | None = None) -> None:
        self._cfg = config or TtsConfig()
        self._lock = threading.Lock()
        self._backend = "disabled"
        self._sapi_local = threading.local()
        self._ps_local = threading.local()
        self._warned_unavailable = False
        self._init_backend()

    @property
    def backend(self) -> str:
        return self._backend

    def _init_backend(self) -> None:
        backend = (self._cfg.backend or "auto").strip().lower()
        if backend == "disabled":
            self._backend = "disabled"
            return

        if backend in ("auto", "sapi"):
            try:
                # win32com 在某些线程中导入/初始化时也可能触发 COM 访问；
                # 先做一次线程级 COM 初始化以避免 “尚未调用 CoInitialize”。
                if not self._ensure_com_initialized():
                    raise RuntimeError("SAPI COM 初始化失败（pythoncom.CoInitialize）。")

                import win32com.client  # type: ignore

                # 预热一次（当前线程的 COM 初始化 + voice 创建）；其它线程会各自创建线程本地 voice。
                try:
                    _ = self._get_sapi_voice()
                except Exception:
                    pass
                self._backend = "sapi"
                logger.info("TTS 后端：SAPI（pywin32）")
                return
            except Exception as exc:  # noqa: BLE001
                if backend == "sapi":
                    logger.warning("SAPI 初始化失败，将禁用 TTS：%s", exc)
                    self._backend = "disabled"
                    return
                logger.info("SAPI 不可用，将尝试 PowerShell TTS：%s", exc)

        if backend in ("auto", "powershell"):
            self._backend = "powershell"
            logger.info("TTS 后端：PowerShell（System.Speech）")
            return

        logger.warning("未知 TTS backend=%s，将禁用 TTS。", backend)
        self._backend = "disabled"

    def speak(self, text: str) -> bool:
        """朗读文本（同步）。成功返回 True，失败返回 False（并在日志中说明）。"""

        text = (text or "").strip()
        if not text:
            return False
        if self._backend == "disabled":
            if not self._warned_unavailable:
                logger.warning("TTS 已禁用或不可用，将仅打印文本，不播放语音。")
                self._warned_unavailable = True
            return False

        with self._lock:
            try:
                if self._backend == "sapi":
                    return self._speak_sapi(text)
                if self._backend == "powershell":
                    return self._speak_powershell(text)
                return False
            except Exception as exc:  # noqa: BLE001
                logger.warning("TTS 播放失败（已忽略）：%s", exc)
                return False

    def stop(self) -> None:
        """尽力停止当前播放（用于“播放/重播”场景）。"""

        if self._backend == "disabled":
            return
        try:
            if self._backend == "sapi":
                voice = self._get_sapi_voice()
                # SVSFPurgeBeforeSpeak = 2：清空队列并中断当前朗读
                voice.Speak("", 2)
                return
            if self._backend == "powershell":
                proc = getattr(self._ps_local, "proc", None)
                if proc is None:
                    return
                try:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=1.0)
                        except Exception:
                            proc.kill()
                finally:
                    self._ps_local.proc = None
        except Exception:
            pass

    def speak_async(self, text: str) -> bool:
        """朗读文本（异步启动）。用于可被“打断并重播”的播放按钮。"""

        text = (text or "").strip()
        if not text:
            return False
        if self._backend == "disabled":
            return False

        try:
            if self._backend == "sapi":
                voice = self._get_sapi_voice()
                # SVSFlagsAsync = 1：异步朗读；这样下一次 stop()/Speak 可立刻打断
                voice.Speak(text, 1)
                return True
            if self._backend == "powershell":
                self.stop()
                text_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
                ps = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$tts = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{text_b64}')); "
                    "$tts.Speak($text);"
                )
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                cmd = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                self._ps_local.proc = proc

                def _killer(p):  # type: ignore[no-untyped-def]
                    try:
                        p.wait(timeout=float(self._cfg.timeout_s))
                    except Exception:
                        try:
                            p.kill()
                        except Exception:
                            pass

                threading.Thread(target=_killer, args=(proc,), daemon=True, name="TtsPsTimeout").start()
                return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("TTS 播放失败（已忽略）：%s", exc)
            return False

    def _ensure_com_initialized(self) -> bool:
        """
        SAPI 基于 COM：每个线程都需要先 CoInitialize，且 COM 对象不能跨线程复用。

        现象：`(-2147221008, '尚未调用 CoInitialize。', ...)`。
        """

        try:
            import pythoncom  # type: ignore
        except Exception:
            return False

        if getattr(self._sapi_local, "com_inited", False):
            return True

        try:
            pythoncom.CoInitialize()
            self._sapi_local.com_inited = True
            return True
        except Exception:
            return False

    def _get_sapi_voice(self):
        voice = getattr(self._sapi_local, "voice", None)
        if voice is not None:
            return voice

        if not self._ensure_com_initialized():
            raise RuntimeError("SAPI COM 初始化失败（pythoncom.CoInitialize）。")

        import win32com.client  # type: ignore

        voice = win32com.client.Dispatch("SAPI.SpVoice")

        # 可选：设置语音
        if self._cfg.voice_name:
            keyword = self._cfg.voice_name.strip().lower()
            try:
                voices = voice.GetVoices()
                picked = None
                for i in range(int(voices.Count)):
                    token = voices.Item(i)
                    desc = str(token.GetDescription()).lower()
                    if keyword in desc:
                        picked = token
                        break
                if picked is not None:
                    voice.Voice = picked
                else:
                    logger.warning("未找到匹配的 SAPI Voice：%s（将使用默认 Voice）", self._cfg.voice_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("设置 SAPI Voice 失败（将使用默认 Voice）：%s", exc)

        if self._cfg.rate is not None:
            try:
                voice.Rate = int(self._cfg.rate)
            except Exception as exc:  # noqa: BLE001
                logger.warning("设置 SAPI 语速失败：%s", exc)
        if self._cfg.volume is not None:
            try:
                voice.Volume = int(self._cfg.volume)
            except Exception as exc:  # noqa: BLE001
                logger.warning("设置 SAPI 音量失败：%s", exc)

        self._sapi_local.voice = voice
        return voice

    def _speak_sapi(self, text: str) -> bool:
        voice = self._get_sapi_voice()
        # 0 = 同步；1 = 异步（此处使用同步更易控流程）
        voice.Speak(text, 0)
        return True

    def _speak_powershell(self, text: str) -> bool:
        # 使用 base64 避免 PowerShell 引号/转义问题
        text_b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            "$tts = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$text = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{text_b64}')); "
            "$tts.Speak($text);"
        )

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        # 强制使用 powershell.exe（Windows PowerShell，.NET Framework），兼容 System.Speech
        cmd = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=float(self._cfg.timeout_s),
            creationflags=creationflags,
        )
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            msg = stderr or stdout or f"returncode={proc.returncode}"
            raise RuntimeError(f"PowerShell TTS 失败：{msg[:500]}")
        return True

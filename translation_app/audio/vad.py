"""
基于 webrtcvad 的低延迟实时分段器（20ms 帧）。

职责：
- 从音频帧队列消费 16kHz/mono/16-bit PCM（每帧 20ms）
- 运行 VAD，按“连续静音超过阈值”切段
- 过滤过短 segment（默认 400ms）以降低噪声触发
"""

from __future__ import annotations

import logging
import queue
from collections import deque
from dataclasses import dataclass
from threading import Event

import webrtcvad

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioFrame:
    """单帧音频（20ms）。"""

    timestamp: float  # 相对会话起点的秒数（单调时间）
    pcm16_bytes: bytes  # 16-bit little-endian PCM，单声道


@dataclass(frozen=True)
class SpeechSegment:
    """一次语音段。"""

    segment_id: int
    start_time: float
    end_time: float
    sample_rate: int
    pcm16_bytes: bytes


class VadSegmenter:
    """
    webrtcvad 分段器：消费 AudioFrame，产出 SpeechSegment。

    设计目标：稳定优先、低延迟切段、异常可读且可恢复。
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        vad_aggressiveness: int = 2,
        silence_ms: int = 600,
        min_segment_ms: int = 1000,
        start_trigger_ms: int = 40,
        pre_speech_ms: int = 200,
        tail_keep_ms: int = 100,
    ) -> None:
        if sample_rate != 16000:
            # webrtcvad 支持 8/16/32/48kHz，这里固定 16k 以匹配需求
            raise ValueError("当前实现要求 sample_rate=16000Hz。")
        if frame_ms not in (10, 20, 30):
            raise ValueError("webrtcvad 仅支持 10/20/30ms 帧。")
        if not (0 <= vad_aggressiveness <= 3):
            raise ValueError("vad_aggressiveness 必须在 0~3。")

        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.vad = webrtcvad.Vad(vad_aggressiveness)

        self.silence_frames_needed = max(1, (silence_ms + frame_ms - 1) // frame_ms)
        self.min_frames = max(1, (min_segment_ms + frame_ms - 1) // frame_ms)
        self.start_trigger_frames = max(1, (start_trigger_ms + frame_ms - 1) // frame_ms)
        self.pre_speech_frames = max(1, (pre_speech_ms + frame_ms - 1) // frame_ms)
        self.tail_keep_frames = max(0, (tail_keep_ms + frame_ms - 1) // frame_ms)

        samples_per_frame = int(sample_rate * frame_ms / 1000)
        self.expected_frame_bytes = samples_per_frame * 2  # int16

    def run(
        self,
        frame_queue: "queue.Queue[AudioFrame]",
        segment_queue: "queue.Queue[SpeechSegment]",
        stop_event: Event,
        pause_event: Event | None = None,
    ) -> None:
        """
        主循环：从 frame_queue 读取帧并分段写入 segment_queue。

        约定：
        - 使用 stop_event 控制退出
        - 队列满时丢弃并打印 warning（稳定优先，不阻塞音频线程）
        """

        segment_id = 0

        pre_buffer: deque[AudioFrame] = deque(maxlen=self.pre_speech_frames)
        in_segment = False
        speech_candidate = 0
        silence_count = 0
        last_speech_index = -1
        frames: list[AudioFrame] = []

        def drain_frames(q: "queue.Queue[AudioFrame]") -> None:
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    return

        def emit_segment(effective_frames: list[AudioFrame]) -> bool:
            nonlocal segment_id
            if len(effective_frames) < self.min_frames:
                return False
            segment_id += 1
            start_time = effective_frames[0].timestamp
            end_time = effective_frames[-1].timestamp + (self.frame_ms / 1000.0)
            pcm16 = b"".join(f.pcm16_bytes for f in effective_frames)
            seg = SpeechSegment(
                segment_id=segment_id,
                start_time=start_time,
                end_time=end_time,
                sample_rate=self.sample_rate,
                pcm16_bytes=pcm16,
            )
            try:
                segment_queue.put_nowait(seg)
                if pause_event is not None:
                    # 对话模式下：一个段结束即暂停继续采集（避免用户输入/回声期间生成新段）
                    pause_event.set()
                return True
            except queue.Full:
                logger.warning("segment_queue 已满，丢弃段 %d（%.2fs~%.2fs）", segment_id, start_time, end_time)
                return False

        while not stop_event.is_set():
            if pause_event is not None and pause_event.is_set():
                # 暂停期间：丢弃所有帧并重置状态，确保恢复后从“干净状态”开始
                drain_frames(frame_queue)
                pre_buffer.clear()
                in_segment = False
                speech_candidate = 0
                silence_count = 0
                last_speech_index = -1
                frames = []
                try:
                    stop_event.wait(0.05)
                except Exception:
                    pass
                continue

            try:
                frame = frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # 基本合法性检查
            if len(frame.pcm16_bytes) != self.expected_frame_bytes:
                logger.warning(
                    "收到异常帧长度：%d bytes（期望 %d），将跳过该帧。",
                    len(frame.pcm16_bytes),
                    self.expected_frame_bytes,
                )
                continue

            try:
                is_speech = self.vad.is_speech(frame.pcm16_bytes, self.sample_rate)
            except Exception as exc:  # noqa: BLE001 - VAD 失败应可恢复
                logger.warning("VAD 判定失败，跳过该帧：%s", exc)
                continue

            if not in_segment:
                pre_buffer.append(frame)
                if is_speech:
                    speech_candidate += 1
                    if speech_candidate >= self.start_trigger_frames:
                        in_segment = True
                        frames = list(pre_buffer)
                        last_speech_index = len(frames) - 1
                        silence_count = 0
                        speech_candidate = 0
                else:
                    speech_candidate = 0
                continue

            # in_segment == True
            frames.append(frame)
            if is_speech:
                silence_count = 0
                last_speech_index = len(frames) - 1
                continue

            silence_count += 1
            if silence_count < self.silence_frames_needed:
                continue

            # 连续静音超过阈值：切段（音频内容尽量不包含过长静音）
            if last_speech_index < 0:
                effective_end = 0
            else:
                effective_end = min(len(frames), last_speech_index + 1 + self.tail_keep_frames)
            effective_frames = frames[:effective_end] if effective_end > 0 else []
            emit_segment(effective_frames)

            # 为下一段做准备：保留最近一小段帧作为 pre-buffer（包含当前静音帧也无妨）
            pre_buffer.clear()
            for keep in frames[-self.pre_speech_frames :]:
                pre_buffer.append(keep)
            in_segment = False
            frames = []
            silence_count = 0
            last_speech_index = -1
            speech_candidate = 0

        # 退出前尽量 flush 最后一段
        if in_segment and frames:
            effective_end = min(len(frames), last_speech_index + 1 + self.tail_keep_frames) if last_speech_index >= 0 else 0
            effective_frames = frames[:effective_end] if effective_end > 0 else []
            emit_segment(effective_frames)

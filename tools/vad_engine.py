"""
VAD 引擎封装 —— 基于 silero-vad 的人声活动检测。

封装 silero-vad 的 VADIterator，提供简化的接口：
  - process_chunk(pcm_bytes) → "speech_start" | "speech_continue" | "speech_end"
  - reset() → 重置状态，新一轮对话开始

输入要求：16kHz, 16bit, mono PCM 原始数据，每帧 512 samples（32ms）。
silero-vad 内部要求帧长为 512 samples（32ms @ 16kHz），
前端发送的 20ms 帧（320 samples）需要在调用前累积到 512 samples。

用法:
    from tools.vad_engine import VADEngine

    engine = VADEngine(silence_threshold_ms=800)
    result = engine.process_chunk(pcm_bytes)
    if result == "speech_end":
        audio = engine.get_accumulated_audio()
        # 送去 STT...
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# silero-vad 要求的帧长（samples），16kHz 下 = 32ms
_SILERO_FRAME_SAMPLES = 512
_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # 16bit


class VADEngine:
    """人声活动检测引擎。

    状态机：
      IDLE → SPEAKING（检测到人声） → IDLE（静音超过阈值）

    属性:
      silence_threshold_ms: 静音持续多少毫秒后判定为 speech_end
    """

    def __init__(self, silence_threshold_ms: int = 800):
        self._silence_threshold_ms = silence_threshold_ms
        self._iterator = None
        self._accumulated_audio = bytearray()
        self._is_speaking = False
        self._silence_duration_ms = 0
        self._init_iterator()

    def _init_iterator(self):
        """初始化 silero-vad 迭代器。"""
        try:
            from silero_vad import VADIterator, load_silero_vad
            model = load_silero_vad()
            self._iterator = VADIterator(
                model,
                sampling_rate=_SAMPLE_RATE,
                threshold=0.5,
                min_silence_duration_ms=self._silence_threshold_ms,
            )
            logger.info("VAD 引擎初始化成功, silence_threshold=%dms", self._silence_threshold_ms)
        except Exception as e:
            logger.error("VAD 引擎初始化失败: %s", e)
            raise RuntimeError(f"VAD 引擎初始化失败: {e}") from e

    def process_chunk(self, pcm_bytes: bytes) -> str:
        """处理一帧 PCM 音频数据。

        Args:
            pcm_bytes: 原始 PCM 字节（16kHz, 16bit, mono），
                       帧长应为 512 samples（1024 字节）。

        Returns:
            "speech_start"  - 从静音进入人声
            "speech_continue" - 持续人声中
            "speech_end"    - 人声结束（静音超过阈值）
            "silence"       - 静音中（无人声）
        """
        if self._iterator is None:
            self._init_iterator()

        # 累积音频数据
        self._accumulated_audio.extend(pcm_bytes)

        # 将 bytes 转为 float32 数组（silero-vad 要求的输入格式）
        import numpy as np
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # 如果帧长不足 512 samples，补零
        if len(samples) < _SILERO_FRAME_SAMPLES:
            padded = np.zeros(_SILERO_FRAME_SAMPLES, dtype=np.float32)
            padded[:len(samples)] = samples
            samples = padded

        # 调用 silero-vad
        speech_dict = self._iterator(samples, return_seconds=False)

        if speech_dict is None:
            # 无事件
            if self._is_speaking:
                return "speech_continue"
            return "silence"

        # 解析 VAD 事件
        if "start" in speech_dict:
            self._is_speaking = True
            logger.debug("VAD: speech_start")
            return "speech_start"

        if "end" in speech_dict:
            self._is_speaking = False
            logger.debug("VAD: speech_end, accumulated=%d bytes", len(self._accumulated_audio))
            return "speech_end"

        if self._is_speaking:
            return "speech_continue"
        return "silence"

    def get_accumulated_audio(self) -> bytes:
        """获取从 speech_start 到 speech_end 之间累积的 PCM 音频数据。"""
        return bytes(self._accumulated_audio)

    def reset(self):
        """重置状态，清空累积音频，准备新一轮对话。"""
        self._accumulated_audio = bytearray()
        self._is_speaking = False
        self._silence_duration_ms = 0
        if self._iterator is not None:
            self._iterator.reset_states()
        logger.debug("VAD: 状态已重置")

    def close(self):
        """释放资源。"""
        self._accumulated_audio = bytearray()
        self._iterator = None
        logger.debug("VAD: 引擎已关闭")

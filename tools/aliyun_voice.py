"""
Aliyun DashScope 语音适配器 -- STT 和 TTS via OpenAI-compatible API.

一次性使用：音频在内存中处理，临时文件立即删除，不做任何持久存储。

配置（cli-config.yaml voice: 段）:
  voice:
    provider: "aliyun"
    aliyun:
      api_key: ""              # 或 DASHSCOPE_API_KEY 环境变量
      stt_model: "sensevoice-v1"
      tts_model: "cosyvoice-v1"
      tts_voice: "longxiaochun"
      tts_format: "mp3"
      base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"

用法:
    from tools.aliyun_voice import aliyun_stt, aliyun_tts, pcm_to_wav

    # STT: 裸 PCM 字节 → 自动加 WAV header → 文字
    result = aliyun_stt(pcm_bytes, "wav")
    if result["success"]:
        print(result["transcript"])

    # 或者手动把 PCM 转成 WAV 再调用
    wav_bytes = pcm_to_wav(pcm_bytes)
    result = aliyun_stt(wav_bytes, "wav")

    # TTS: 文字 → 音频字节
    result = aliyun_tts("你好世界")
    if result["success"]:
        audio_data = result["audio_data"]  # MP3 bytes
"""

from __future__ import annotations

import logging
import os
import re
import struct
import tempfile
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认值
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_STT_MODEL = "sensevoice-v1"
DEFAULT_TTS_MODEL = "cosyvoice-v1"
DEFAULT_TTS_VOICE = "longxiaochun"
DEFAULT_TTS_FORMAT = "mp3"
MAX_STT_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB
MAX_TTS_TEXT_LENGTH = 4000

# PCM 参数（与前端 AudioWorklet 保持一致）
PCM_SAMPLE_RATE = 16000
PCM_BITS_PER_SAMPLE = 16
PCM_CHANNELS = 1


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------


def _load_voice_config() -> dict:
    """从 cli-config.yaml 加载 voice: 段配置。"""
    try:
        from hermes_cli.config import load_config
        return load_config().get("voice", {})
    except Exception:
        return {}


def _get_aliyun_config() -> dict:
    """获取 aliyun 子配置。"""
    voice_cfg = _load_voice_config()
    return voice_cfg.get("aliyun", {})


def _get_api_key() -> str:
    """获取 DashScope API Key，优先配置文件，其次环境变量。"""
    cfg = _get_aliyun_config()
    key = cfg.get("api_key", "")
    if key:
        return key
    return os.getenv("DASHSCOPE_API_KEY", "")


def _check_available() -> bool:
    """检查 Aliyun 语音服务是否可用（API Key + openai 包）。"""
    if not _get_api_key():
        return False
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# PCM → WAV 转换
# ---------------------------------------------------------------------------


def pcm_to_wav(pcm_data: bytes,
               sample_rate: int = PCM_SAMPLE_RATE,
               bits_per_sample: int = PCM_BITS_PER_SAMPLE,
               channels: int = PCM_CHANNELS) -> bytes:
    """将裸 PCM 数据加上 WAV 文件头，生成标准 WAV 格式。

    DashScope sensevoice-v1 的 OpenAI-compatible 端点需要标准音频文件格式，
    裸 PCM 无法直接识别。此函数在内存中完成转换，不产生磁盘 IO。

    Args:
        pcm_data: 裸 PCM 字节（16kHz, 16bit, mono）
        sample_rate: 采样率，默认 16000
        bits_per_sample: 位深，默认 16
        channels: 声道数，默认 1

    Returns:
        带 WAV header 的完整音频字节
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm_data)

    # WAV header: 44 字节
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',                          # ChunkID
        36 + data_size,                   # ChunkSize
        b'WAVE',                          # Format
        b'fmt ',                          # Subchunk1ID
        16,                               # Subchunk1Size (PCM = 16)
        1,                                # AudioFormat (1 = PCM)
        channels,                         # NumChannels
        sample_rate,                      # SampleRate
        byte_rate,                        # ByteRate
        block_align,                      # BlockAlign
        bits_per_sample,                  # BitsPerSample
        b'data',                          # Subchunk2ID
        data_size,                        # Subchunk2Size
    )

    return header + pcm_data


# ---------------------------------------------------------------------------
# Markdown 清理（和 CLI voice mode / tts_tool.py 保持一致）
# ---------------------------------------------------------------------------

def _strip_markdown_for_tts(text: str) -> str:
    """去除 markdown 格式，让 TTS 读出来更自然。"""
    text = re.sub(r'```[\s\S]*?```', ' ', text)          # 代码块
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # 链接
    text = re.sub(r'https?://\S+', '', text)              # URL
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)          # 加粗
    text = re.sub(r'\*(.+?)\*', r'\1', text)              # 斜体
    text = re.sub(r'`(.+?)`', r'\1', text)                # 行内代码
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # 标题
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)  # 列表
    text = re.sub(r'---+', '', text)                      # 分隔线
    text = re.sub(r'\n{3,}', '\n\n', text)               # 多余换行
    return text.strip()


# ---------------------------------------------------------------------------
# STT: 语音转文字
# ---------------------------------------------------------------------------


def aliyun_stt(audio_data: bytes, audio_format: str = "wav") -> Dict[str, Any]:
    """使用 Aliyun DashScope 原生流式 STT 转写音频。

    这里不再走 OpenAI-compatible `/audio/transcriptions`，
    而是改用 DashScope SDK 的 `Recognition.start/send_audio_frame/stop`。

    Args:
        audio_data: 原始音频字节。当前优先支持 WAV / PCM。
        audio_format: 音频格式扩展名（wav、pcm），默认 wav。

    Returns:
        {"success": True, "transcript": "转写文字"}
        或 {"success": False, "error": "错误描述"}
    """
    api_key = _get_api_key()
    if not api_key:
        return {"success": False, "error": "DASHSCOPE_API_KEY 未配置"}

    if len(audio_data) > MAX_STT_AUDIO_BYTES:
        return {"success": False, "error": f"音频过大: {len(audio_data) / 1024 / 1024:.1f}MB (上限 25MB)"}

    if not audio_data:
        return {"success": False, "error": "音频数据为空"}

    cfg = _get_aliyun_config()
    model = cfg.get("stt_model", DEFAULT_STT_MODEL)

    # Recognition 流式接口更适合本地短音频直传。
    # WebSocket 语音链路当前产物是 16kHz / mono / 16bit PCM 封装成 WAV。
    if audio_format.lower() not in {"wav", "pcm"}:
        return {"success": False, "error": f"Aliyun STT 暂不支持当前直传格式: {audio_format}"}

    try:
        import dashscope
        from dashscope.audio.asr.recognition import Recognition, RecognitionCallback

        dashscope.api_key = api_key

        # WAV 需要剥掉 44 字节头，Recognition 流式接口送原始 PCM 更稳。
        pcm_payload = audio_data
        if audio_format.lower() == "wav" and len(audio_data) > 44 and audio_data[:4] == b"RIFF":
            pcm_payload = audio_data[44:]

        class _Callback(RecognitionCallback):
            def __init__(self):
                self.sentences = []
                self.error_message = ""

            def on_event(self, result):
                try:
                    sentence = result.get_sentence()
                    logger.info("Aliyun Recognition on_event sentence=%r", sentence)
                    if isinstance(sentence, dict):
                        from dashscope.audio.asr.recognition import RecognitionResult
                        if RecognitionResult.is_sentence_end(sentence):
                            text = sentence.get("text")
                            if text:
                                self.sentences.append(text)
                    elif isinstance(sentence, list):
                        from dashscope.audio.asr.recognition import RecognitionResult
                        for item in sentence:
                            logger.info("Aliyun Recognition on_event sentence_item=%r", item)
                            if isinstance(item, dict) and RecognitionResult.is_sentence_end(item):
                                text = item.get("text")
                                if text:
                                    self.sentences.append(text)
                except Exception as exc:
                    self.error_message = f"Aliyun Recognition 回调解析失败: {exc}"

            def on_error(self, result):
                try:
                    self.error_message = str(result)
                except Exception:
                    self.error_message = "Aliyun Recognition on_error"

        callback = _Callback()
        recognition = Recognition(
            model=model,
            callback=callback,
            format="pcm",
            sample_rate=16000,
        )
        recognition.start()

        # 分片发送，避免一次过大。
        frame_size = 3200  # 约 100ms @ 16kHz, 16bit, mono
        for i in range(0, len(pcm_payload), frame_size):
            recognition.send_audio_frame(pcm_payload[i:i + frame_size])
        recognition.stop()

        transcript = "".join(callback.sentences).strip()
        if callback.error_message and not transcript:
            return {"success": False, "error": callback.error_message}
        if not transcript:
            return {"success": False, "error": "Aliyun STT 未返回可用转写文本"}

        logger.info("Aliyun STT 转写成功: %d 字节音频 → %d 字文字", len(audio_data), len(transcript))
        return {"success": True, "transcript": transcript}

    except Exception as e:
        logger.warning("Aliyun STT 失败: %s", e)
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# TTS: 文字转语音
# ---------------------------------------------------------------------------


def aliyun_tts(text: str) -> Dict[str, Any]:
    """使用 Aliyun DashScope CosyVoice TTS 生成语音音频。

    通过 DashScope 原生 SDK（SpeechSynthesizer）调用，
    不再使用 OpenAI-compatible 端点（该端点不支持 /v1/audio/speech）。

    Args:
        text: 要转换的文字（最长 4000 字符，超出自动截断）

    Returns:
        {"success": True, "audio_data": bytes, "format": "mp3"}
        或 {"success": False, "error": "错误描述"}

    音频数据仅在内存中返回，不做任何持久存储。
    """
    api_key = _get_api_key()
    if not api_key:
        return {"success": False, "error": "DASHSCOPE_API_KEY 未配置"}

    if not text or not text.strip():
        return {"success": False, "error": "文字为空"}

    # 合并 voice.aliyun 和 tts.aliyun 配置，tts 优先覆盖 voice
    voice_cfg = _load_voice_config()
    voice_aliyun = voice_cfg.get("aliyun", {})
    try:
        from hermes_cli.config import load_config
        tts_cfg = load_config().get("tts", {})
        tts_aliyun = tts_cfg.get("aliyun", {})
    except Exception:
        tts_aliyun = {}

    merged = {}
    merged.update(voice_aliyun)
    merged.update(tts_aliyun)

    model = merged.get("tts_model", DEFAULT_TTS_MODEL)
    voice = merged.get("tts_voice", DEFAULT_TTS_VOICE)
    fmt = merged.get("tts_format", DEFAULT_TTS_FORMAT)

    # 清理 markdown 并截断过长文本
    tts_text = _strip_markdown_for_tts(text)[:MAX_TTS_TEXT_LENGTH]
    if not tts_text:
        return {"success": False, "error": "清理后文字为空"}

    # 格式映射：用户配置 → DashScope AudioFormat 枚举
    from dashscope.audio.tts_v2 import AudioFormat
    _FORMAT_MAP = {
        "mp3": AudioFormat.MP3_22050HZ_MONO_256KBPS,
        "wav": AudioFormat.WAV_22050HZ_MONO_16BIT,
        "pcm": AudioFormat.PCM_22050HZ_MONO_16BIT,
        "opus": AudioFormat.OGG_OPUS_24KHZ_MONO_32KBPS,
    }
    audio_format = _FORMAT_MAP.get(fmt, AudioFormat.MP3_22050HZ_MONO_256KBPS)

    try:
        # DashScope SDK 在 import 时读取环境变量 DASHSCOPE_API_KEY，
        # 同时设置 dashscope.api_key 属性确保已导入的模块也能感知
        import os as _os
        import dashscope
        _old_env_key = _os.environ.get("DASHSCOPE_API_KEY", "")
        _old_attr_key = dashscope.api_key
        _os.environ["DASHSCOPE_API_KEY"] = api_key
        dashscope.api_key = api_key

        from dashscope.audio.tts_v2 import SpeechSynthesizer

        synthesizer = SpeechSynthesizer(
            model=model,
            voice=voice,
            format=audio_format,
        )

        audio_data = synthesizer.call(tts_text)

        # 恢复原始状态
        if _old_env_key:
            _os.environ["DASHSCOPE_API_KEY"] = _old_env_key
        else:
            _os.environ.pop("DASHSCOPE_API_KEY", None)
        dashscope.api_key = _old_attr_key

        if audio_data is None:
            return {"success": False, "error": "DashScope TTS 返回空音频数据"}

        logger.info("Aliyun TTS 生成成功: %d 字文字 → %d 字节音频(%s, model=%s, voice=%s)",
                     len(tts_text), len(audio_data), fmt, model, voice)
        return {"success": True, "audio_data": audio_data, "format": fmt}

    except Exception as e:
        logger.warning("Aliyun TTS 失败: %s", e)
        return {"success": False, "error": str(e)}
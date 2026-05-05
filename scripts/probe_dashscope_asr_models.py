#!/usr/bin/env python3
"""探测当前 DashScope 账号下可用的 Recognition ASR 模型。

用途：
1. 逐个尝试候选模型名；
2. 用最小 PCM 音频帧触发一次 `Recognition.start()` / `send_audio_frame()` / `stop()`；
3. 输出每个模型是成功、超时还是具体报错；
4. 帮助确认当前账号/地域到底支持哪个实时识别模型。

运行方式：
    python3 scripts/probe_dashscope_asr_models.py

说明：
- 这个脚本只做探测，不改配置；
- 优先从 Hermes 配置里读取阿里云 API Key；
- 如果配置里没有，再回退读 `DASHSCOPE_API_KEY` 环境变量；
- 发送的是极短静音 PCM，目的只是验证“模型可用性”，不是验证识别准确率。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dashscope
from dashscope.audio.asr.recognition import Recognition, RecognitionCallback

from hermes_cli.config import load_config


CANDIDATE_MODELS = [
    "paraformer-realtime-v1",
    "paraformer-realtime-v2",
    "paraformer-realtime-8k-v1",
    "paraformer-realtime-8k-v2",
    "paraformer-v1",
    "paraformer-16k-1",
    "qwen3-asr",
    "sensevoice-v1",
]

# 16kHz / mono / 16bit PCM，约 200ms 静音
SILENCE_PCM_16K = b"\x00\x00" * 3200


@dataclass
class ProbeResult:
    model: str
    ok: bool
    transcript: str
    error: str
    request_id: str


class _ProbeCallback(RecognitionCallback):
    """收集 Recognition 回调结果。"""

    def __init__(self) -> None:
        self._done = threading.Event()
        self.sentences: list[str] = []
        self.error_message = ""
        self.request_id = ""

    def on_open(self) -> None:
        return None

    def on_close(self) -> None:
        # 某些情况只触发 close，不触发 complete
        self._done.set()

    def on_complete(self) -> None:
        self._done.set()

    def on_error(self, result) -> None:
        self.request_id = getattr(result, "request_id", "") or ""
        self.error_message = str(result)
        self._done.set()

    def on_event(self, result) -> None:
        self.request_id = getattr(result, "request_id", "") or self.request_id
        try:
            if result.is_sentence_end():
                sentence = result.get_sentence()
                if sentence and sentence.get("text"):
                    self.sentences.append(sentence["text"])
        except Exception as exc:  # noqa: BLE001
            self.error_message = f"回调解析失败: {exc}"
            self._done.set()

    def wait(self, timeout: float = 8.0) -> bool:
        return self._done.wait(timeout)


def _load_dashscope_key() -> str:
    config = load_config()
    voice_aliyun = (config.get("voice") or {}).get("aliyun") or {}
    tts_aliyun = (config.get("tts") or {}).get("aliyun") or {}
    model_aliyun = (config.get("model") or {}) if isinstance(config.get("model"), dict) else {}
    return (
        voice_aliyun.get("api_key")
        or tts_aliyun.get("api_key")
        or model_aliyun.get("api_key")
        or ""
    )


def probe_model(model: str, *, api_key: str) -> ProbeResult:
    callback = _ProbeCallback()
    try:
        recognition = Recognition(
            model=model,
            callback=callback,
            format="pcm",
            sample_rate=16000,
        )
        recognition.start()
        recognition.send_audio_frame(SILENCE_PCM_16K)
        recognition.stop()

        callback.wait(timeout=8.0)
        transcript = "".join(callback.sentences).strip()
        ok = not callback.error_message
        return ProbeResult(
            model=model,
            ok=ok,
            transcript=transcript,
            error=callback.error_message,
            request_id=callback.request_id,
        )
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(
            model=model,
            ok=False,
            transcript="",
            error=str(exc),
            request_id="",
        )


def main() -> int:
    api_key = _load_dashscope_key()
    if not api_key:
        print("未找到 DashScope API Key。请先检查 [../.hermes/config.yaml](../.hermes/config.yaml) 中的 aliyun 配置。")
        return 1

    dashscope.api_key = api_key

    print("开始探测 DashScope Recognition 可用模型...\n")
    success_models: list[str] = []

    for model in CANDIDATE_MODELS:
        print(f"[probe] {model}")
        result = probe_model(model, api_key=api_key)
        if result.ok:
            success_models.append(model)
            transcript_text = result.transcript or "<空转写>"
            print(f"  OK  request_id={result.request_id or '-'} transcript={transcript_text}")
        else:
            print(f"  ERR request_id={result.request_id or '-'} error={result.error}")
        time.sleep(0.2)

    print("\n探测完成。")
    if success_models:
        print("可用模型：")
        for model in success_models:
            print(f"- {model}")
        return 0

    print("没有探测到可用模型。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

# 语音对话 VAD 驱动实时交互 — 设计方案

日期：2026-04-29

## 背景

当前 push-to-talk（按住说话）语音交互模式存在问题，用户反馈"还是不行"。经排查，根因是 `voice.aliyun` 配置段缺失导致 API Key 为空，STT/TTS 第一步就失败。但即使修复配置，push-to-talk 交互模式本身也不够自然。

## 目标

将语音交互从 push-to-talk 改为 **VAD 驱动的实时对话模式**：
- 点击麦克风进入持续监听，无需按住
- 后端用 silero-vad 检测人声边界
- 人声结束后自动触发 STT → Agent → TTS
- AI 回复期间用户可随时开口打断
- 前端用 AudioWorklet 直采 PCM，替代 MediaRecorder

## 状态机

```
        点击麦克风
            │
            ▼
        ┌──────────┐
        │ LISTENING│  ← 持续接收 PCM，VAD 实时分析
        └────┬─────┘
             │ VAD 检测到静音 >= 800ms
             ▼
        ┌──────────┐
        │ THINKING │  ← STT + Agent + TTS
        └────┬─────┘
             │ 首个 token 就绪
             ▼
        ┌──────────┐
        │ SPEAKING │  ← 播放 TTS + 流式文字
        └────┬─────┘
             │ TTS 播放完毕 → 直接回到 LISTENING
             │
             │ 用户开口打断 → 回到 LISTENING（清空队列）
```

IDLE 仅作为初始状态（连接建立后、尚未点击麦克风时），进入语音模式后只有 LISTENING / THINKING / SPEAKING 三个状态循环。

## WebSocket 协议

**客户端 → 服务端：**
- `{"type": "voice_start"}` — 进入语音模式，开始持续推送音频
- `{"type": "voice_stop"}` — 退出语音模式
- 二进制帧：PCM 音频数据（16kHz, 16bit, mono），每 20ms 一帧（640 字节）

**服务端 → 客户端：**
- `{"type": "vad", "status": "speech_start" | "speech_end"}` — VAD 事件
- `{"type": "transcription", "text": "..."}` — STT 转写结果
- `{"type": "token", "delta": "..."}` — Agent 文本增量
- `{"type": "done", "final": "..."}` — 本轮对话结束
- `{"type": "interrupted"}` — 被打断通知
- `{"type": "error", "error": "..."}` — 错误
- 二进制帧：TTS 音频（MP3 bytes）

## 涉及文件

### 新增

| 文件 | 职责 |
|------|------|
| `tools/vad_engine.py` | VAD 引擎封装，基于 silero-vad |
| `web/src/lib/audioCapture.ts` | AudioWorklet PCM 采集器 |
| `web/public/pcm-processor.js` | AudioWorkletProcessor 实现 |

### 修改

| 文件 | 改动 |
|------|------|
| `hermes_cli/web_server.py` | 重写 `/api/voice/ws` 端点 |
| `tools/aliyun_voice.py` | STT 输入格式从 webm 改为 wav |
| `web/src/lib/voiceClient.ts` | 重写，对接新协议 |
| `web/src/pages/SessionsPage.tsx` | 简化 UI，点击即进入语音模式 |

## 后端核心流程

```
voice_start → 创建 VAD 引擎实例，进入 LISTENING
    │
    ├── 收到 PCM 二进制帧 → vad_engine.process_chunk()
    │       │
    │       ├── speech_start → 标记语音开始，开始累积音频
    │       ├── speech_continue → 继续累积
    │       └── speech_end（静音 >= 800ms）→ 触发处理
    │               │
    │               ├── 如果在 SPEAKING 状态 → 打断！
    │               │     ├── 取消 Agent task
    │               │     ├── 发送 {"type": "interrupted"}
    │               │     └── 清空 TTS 队列
    │               │
    │               └── 累积的 PCM → 加 WAV header → aliyun_stt()
    │                       │
    │                       ├── STT 成功 → Agent → TTS 流式输出
    │                       └── STT 失败 → {"type": "error", ...}
    │
    └── voice_stop → 清理资源，关闭连接
```

## VAD 引擎

- 模型：silero-vad（onnx 格式）
- 输入：每 20ms 一帧，640 字节（16000 * 2 * 0.02）
- 输出：speech_start / speech_continue / speech_end
- 静音阈值：800ms

## 前端 PCM 采集

- 采样率：16000 Hz，位深：16bit，声道：mono
- AudioWorklet 每次输出 128 samples（8ms），前端累积到 320 samples（20ms）再发送
- 每秒 50 帧，约 31.25 KB/s 上行带宽

## 错误处理

| 场景 | 处理 |
|------|------|
| 麦克风权限拒绝 | toast 提示，状态保持 idle |
| WebSocket 断开 | 自动重连（2s），重连后重建 VAD 引擎 |
| STT 失败 | 发送 error 事件，回到 LISTENING |
| TTS 失败 | 仅影响语音，文本正常推送 |
| Agent 异常 | 发送 error 事件，回到 LISTENING |
| VAD 模型加载失败 | 拒绝进入语音模式 |
| 说话太短（< 0.5s） | 丢弃，不触发 STT |
| 说话太长（> 15s） | 强制截断触发 STT |
| TTS 队列堆积（> 10） | 跳过中间帧，保留最新 3 帧 |

## 打断机制

- 使用 `asyncio.Task.cancel()` 取消 Agent
- 清空 TTS 播放队列
- VAD 引擎 `reset()` 避免残留状态
- 前端 `source.stop()` 停止当前音频播放

/**
 * voiceClient —— 浏览器端 VAD 驱动的实时语音对话客户端
 *
 * 职责：
 *   1. 建立与后端 /api/voice/ws 的 WebSocket 长连接
 *   2. 通过 AudioWorklet 采集 PCM 音频并持续推送到服务端
 *   3. 接收并播放后端返回的 TTS 音频（MP3 格式）
 *   4. 管理语音对话状态机：idle → listening → thinking → speaking → listening
 *
 * 交互模式：
 *   点击麦克风进入语音模式，无需按住。VAD 自动检测人声边界，
 *   人声结束后自动触发 STT + Agent + TTS。AI 回复期间可随时开口打断。
 *
 * 用法：
 *   const vc = createVoiceClient();
 *   vc.subscribe((event) => { ... });
 *   vc.start();   // 进入语音模式
 *   vc.stop();    // 退出语音模式
 *   vc.close();   // 销毁
 *
 * 协议（与后端 web_server.py /api/voice/ws 对应）：
 *   客户端 → 服务端（JSON 文本帧）:
 *     {"type": "voice_start"}
 *     {"type": "voice_stop"}
 *
 *   客户端 → 服务端（二进制帧）:
 *     PCM 音频数据（16kHz, 16bit, mono），每 20ms 一帧
 *
 *   服务端 → 客户端（JSON 文本帧）:
 *     {"type": "status", "status": "listening|thinking|speaking|idle|error"}
 *     {"type": "vad", "status": "speech_start|speech_end"}
 *     {"type": "transcription", "text": "…"}
 *     {"type": "token", "delta": "…"}
 *     {"type": "done", "final": "…"}
 *     {"type": "interrupted"}
 *     {"type": "error", "error": "…"}
 *
 *   服务端 → 客户端（二进制帧）:
 *     TTS 音频数据（MP3 bytes）
 */

import { createAudioCapture, type AudioCapture } from "./audioCapture";

export type VoiceStatus = "idle" | "listening" | "thinking" | "speaking" | "error";

export type VoiceEvent =
  | { type: "status"; status: VoiceStatus }
  | { type: "vad"; status: "speech_start" | "speech_end" }
  | { type: "transcription"; text: string }
  | { type: "token"; delta: string }
  | { type: "done"; final: string }
  | { type: "interrupted" }
  | { type: "error"; error: string }
  | { type: "closed" };

type Listener = (event: VoiceEvent) => void;

export interface VoiceClient {
  /** 订阅事件 */
  subscribe(fn: Listener): () => void;
  /** 进入语音模式（开始 PCM 采集 + VAD 检测） */
  start(): Promise<void>;
  /** 退出语音模式 */
  stop(): void;
  /** 销毁客户端 */
  close(): void;
  /** 当前状态 */
  getStatus(): VoiceStatus;
}

export function createVoiceClient(): VoiceClient {
  const listeners = new Set<Listener>();
  let ws: WebSocket | null = null;
  let audioCapture: AudioCapture | null = null;
  let status: VoiceStatus = "idle";
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  let closed = false;
  let voiceActive = false;

  // ---- 内部方法 ----

  function notify(event: VoiceEvent) {
    if (event.type === "status") {
      status = event.status;
    }
    for (const fn of listeners) {
      try {
        fn(event);
      } catch {
        // 忽略单个监听器的异常
      }
    }
  }

  function buildWsUrl(): string {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}/api/voice/ws`;
  }

  function connect() {
    if (closed) return;
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const url = buildWsUrl();
    ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      notify({ type: "status", status: "idle" });
      // 如果之前在语音模式，重连后自动恢复
      if (voiceActive) {
        ws?.send(JSON.stringify({ type: "voice_start" }));
      }
    };

    ws.onmessage = (event: MessageEvent) => {
      // 二进制帧：TTS 音频数据
      if (event.data instanceof ArrayBuffer) {
        playAudioChunk(event.data);
        return;
      }

      // 文本帧：JSON 控制消息
      if (typeof event.data === "string") {
        try {
          const msg = JSON.parse(event.data) as VoiceEvent;
          notify(msg);
        } catch {
          // 忽略解析失败
        }
      }
    };

    ws.onclose = () => {
      if (!closed) {
        reconnectTimer = setTimeout(connect, 2000);
      }
      notify({ type: "closed" });
    };

    ws.onerror = () => {
      // onclose 会紧随其后触发，重连逻辑在 onclose 中处理
    };
  }

  // ---- 音频播放 ----

  let audioContext: AudioContext | null = null;
  let pendingAudioChunks: ArrayBuffer[] = [];
  let isPlaying = false;
  let currentSource: AudioBufferSourceNode | null = null;

  function getAudioContext(): AudioContext {
    if (!audioContext || audioContext.state === "closed") {
      audioContext = new AudioContext();
      console.info("[voice] playback AudioContext created, state=", audioContext.state, "sampleRate=", audioContext.sampleRate);
    }
    return audioContext;
  }

  async function ensureAudioContextReady(ctx: AudioContext): Promise<void> {
    if (ctx.state === "suspended") {
      console.info("[voice] playback AudioContext suspended, resuming...");
      await ctx.resume();
      console.info("[voice] playback AudioContext resumed, state=", ctx.state);
    }
  }

  async function playAudioChunk(arrayBuffer: ArrayBuffer) {
    pendingAudioChunks.push(arrayBuffer);
    console.info("[voice] received audio chunk:", arrayBuffer.byteLength, "bytes, queue=", pendingAudioChunks.length);
    if (isPlaying) return;

    isPlaying = true;
    try {
      while (pendingAudioChunks.length > 0) {
        const chunk = pendingAudioChunks.shift()!;
        if (chunk.byteLength === 0) {
          continue;
        }
        const ctx = getAudioContext();
        await ensureAudioContextReady(ctx);
        console.info("[voice] decoding audio chunk:", chunk.byteLength, "bytes, ctx.state=", ctx.state);
        const audioBuffer = await ctx.decodeAudioData(chunk.slice(0));
        console.info("[voice] decoded audio buffer:", audioBuffer.duration, "sec, channels=", audioBuffer.numberOfChannels, "sampleRate=", audioBuffer.sampleRate);
        currentSource = ctx.createBufferSource();
        currentSource.buffer = audioBuffer;
        currentSource.connect(ctx.destination);
        currentSource.start(0);
        console.info("[voice] audio playback started");

        // 等待播放完成
        await new Promise<void>((resolve) => {
          currentSource!.onended = () => {
            console.info("[voice] audio playback ended");
            currentSource = null;
            resolve();
          };
        });
      }
    } catch (err) {
      console.error("[voice] audio playback failed:", err);
    } finally {
      isPlaying = false;
      currentSource = null;
      // 如果播放期间又有新的 chunk 进来，继续播放
      if (pendingAudioChunks.length > 0) {
        void playAudioChunk(new ArrayBuffer(0));
      }
    }
  }

  function stopAudioPlayback() {
    if (currentSource) {
      try {
        currentSource.stop();
      } catch {
        // 可能已经停止了
      }
      currentSource = null;
    }
    pendingAudioChunks = [];
    isPlaying = false;
  }

  // ---- 语音模式控制 ----

  async function start(): Promise<void> {
    if (closed || voiceActive) return;

    // 确保 WebSocket 已连接
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      connect();
      // 等待连接建立
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error("WebSocket 连接超时")), 5000);
        const check = () => {
          if (ws?.readyState === WebSocket.OPEN) {
            clearTimeout(timeout);
            resolve();
          } else if (ws?.readyState === WebSocket.CLOSED || ws?.readyState === WebSocket.CLOSING) {
            clearTimeout(timeout);
            reject(new Error("WebSocket 连接失败"));
          } else {
            setTimeout(check, 100);
          }
        };
        check();
      });
    }

    // 创建 PCM 采集器
    audioCapture = await createAudioCapture((pcmBytes: ArrayBuffer) => {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(pcmBytes);
      }
    });

    // 发送 voice_start
    ws!.send(JSON.stringify({ type: "voice_start" }));

    // 开始采集
    await audioCapture.start();
    voiceActive = true;
  }

  function stop(): void {
    if (!voiceActive) return;

    // 停止 PCM 采集
    if (audioCapture) {
      audioCapture.stop();
      audioCapture = null;
    }

    // 停止音频播放
    stopAudioPlayback();

    // 发送 voice_stop
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "voice_stop" }));
    }

    voiceActive = false;
    notify({ type: "status", status: "idle" });
  }

  function close(): void {
    closed = true;

    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }

    stop();

    if (audioContext) {
      audioContext.close().catch(() => {});
      audioContext = null;
    }

    if (ws) {
      ws.close();
      ws = null;
    }

    notify({ type: "closed" });
  }

  // ---- 初始化 ----
  connect();

  return {
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    start,
    stop,
    close,
    getStatus: () => status,
  };
}

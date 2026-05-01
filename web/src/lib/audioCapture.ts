/**
 * audioCapture —— AudioWorklet PCM 采集器（带降采样）
 *
 * 替代 MediaRecorder，直接采集 16kHz/16bit/mono 原始 PCM 数据。
 * 自动检测 AudioContext 实际采样率，如果不是 16kHz 则通知 PCMProcessor
 * 在 AudioWorklet 内做线性插值降采样。
 *
 * 每 20ms（320 samples = 640 bytes @ 16kHz）回调一次 onPcmData。
 *
 * 用法：
 *   const capture = await createAudioCapture((pcmBytes: ArrayBuffer) => {
 *     ws.send(pcmBytes);
 *   });
 *   capture.start();
 *   capture.stop();
 */

const TARGET_SAMPLE_RATE = 16000;
const CHUNK_MS = 20; // 每 20ms 发送一帧
const CHUNK_SAMPLES = (TARGET_SAMPLE_RATE * CHUNK_MS) / 1000; // 320 samples
const CHUNK_BYTES = CHUNK_SAMPLES * 2; // 640 bytes (16bit)

export interface AudioCapture {
  /** 开始采集 */
  start(): Promise<void>;
  /** 停止采集 */
  stop(): void;
  /** 是否正在采集 */
  isActive(): boolean;
}

export async function createAudioCapture(
  onPcmData: (pcmBytes: ArrayBuffer) => void
): Promise<AudioCapture> {
  let audioContext: AudioContext | null = null;
  let mediaStream: MediaStream | null = null;
  let workletNode: AudioWorkletNode | null = null;
  let active = false;

  // PCM 累积缓冲区：凑够 20ms 再发送
  let pcmBuffer = new Uint8Array(0);

  function flushBuffer() {
    while (pcmBuffer.length >= CHUNK_BYTES) {
      const chunk = pcmBuffer.slice(0, CHUNK_BYTES);
      pcmBuffer = pcmBuffer.slice(CHUNK_BYTES);
      onPcmData(chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength));
    }
  }

  async function start(): Promise<void> {
    if (active) return;

    // 获取麦克风权限
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        // 请求 16kHz，但浏览器可能不支持，实际采样率由 AudioContext 决定
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    // 创建 AudioContext —— 浏览器会选择支持的采样率
    // Chrome macOS 默认 44.1kHz 或 48kHz，不一定支持 16kHz
    audioContext = new AudioContext();

    const actualSampleRate = audioContext.sampleRate;
    console.log(`[audioCapture] AudioContext sampleRate: ${actualSampleRate}Hz (target: ${TARGET_SAMPLE_RATE}Hz)`);
    const needsResample = actualSampleRate !== TARGET_SAMPLE_RATE;
    if (needsResample) {
      console.log(`[audioCapture] Will resample ${actualSampleRate}Hz → ${TARGET_SAMPLE_RATE}Hz in AudioWorklet`);
    }

    // 加载 AudioWorklet 模块
    await audioContext.audioWorklet.addModule('/pcm-processor.js');

    // 创建 AudioWorkletNode
    workletNode = new AudioWorkletNode(audioContext, 'pcm-processor');

    // 通知 PCMProcessor 实际采样率（用于降采样）
    workletNode.port.postMessage({
      type: 'config',
      sampleRate: actualSampleRate,
    });

    // 接收 PCM 数据（降采样后每帧约 46-58 samples @ 16kHz，取决于输入采样率）
    workletNode.port.onmessage = (event: MessageEvent) => {
      if (!active) return;

      const int16Buffer = event.data as ArrayBuffer;
      const newBytes = new Uint8Array(int16Buffer);

      // 拼接到累积缓冲区
      const merged = new Uint8Array(pcmBuffer.length + newBytes.length);
      merged.set(pcmBuffer, 0);
      merged.set(newBytes, pcmBuffer.length);
      pcmBuffer = merged;

      // 凑够 20ms 就发送
      flushBuffer();
    };

    // 连接音频图：麦克风 → AudioWorkletNode
    const source = audioContext.createMediaStreamSource(mediaStream);
    source.connect(workletNode);
    // AudioWorkletNode 不需要连接到 destination（不播放）

    active = true;
  }

  function stop(): void {
    active = false;

    // 断开并清理
    if (workletNode) {
      workletNode.port.onmessage = null;
      workletNode.disconnect();
      workletNode = null;
    }

    if (audioContext) {
      audioContext.close().catch(() => {});
      audioContext = null;
    }

    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }

    pcmBuffer = new Uint8Array(0);
  }

  function isActive(): boolean {
    return active;
  }

  return { start, stop, isActive };
}

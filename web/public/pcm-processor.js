/**
 * PCM Processor —— AudioWorklet 处理器（带降采样）
 *
 * 从麦克风采集原始 PCM 音频数据，自动降采样到 16kHz/16bit/mono。
 *
 * 浏览器 AudioContext 可能不支持 16kHz（Chrome macOS 默认 44.1kHz/48kHz），
 * 本处理器在 AudioWorklet 内做线性插值降采样，确保输出始终是 16kHz PCM。
 *
 * 如果 AudioContext 实际采样率恰好是 16kHz，则直接输出不做降采样。
 *
 * 配置方式：主线程通过 MessagePort 发送
 *   { type: 'config', sampleRate: <actualAudioContextSampleRate> }
 *
 * 输出：Int16Array 的 ArrayBuffer，通过 MessagePort 发送给主线程。
 * 主线程负责累积到 20ms（320 samples = 640 bytes @ 16kHz）再通过 WebSocket 发送。
 */

const TARGET_RATE = 16000;

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.inputRate = TARGET_RATE; // 默认 16kHz，将通过 config 消息更新
    this.needsResample = false;

    // 降采样状态
    this.inputBuf = new Float32Array(512); // 预分配缓冲区
    this.inputBufLen = 0;                   // 缓冲区中有效样本数
    this.resamplePos = 0;                   // 降采样的分数位置

    this.port.onmessage = (e) => {
      if (e.data.type === 'config' && e.data.sampleRate) {
        this.inputRate = e.data.sampleRate;
        this.needsResample = this.inputRate !== TARGET_RATE;
      }
    };
  }

  _floatToInt16(float32) {
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const s = Math.max(-1, Math.min(1, float32[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16;
  }

  process(inputs, _outputs, _parameters) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const channelData = input[0]; // Float32Array, 128 samples @ inputRate
    if (!channelData) return true;

    // ---- 直接路径：AudioContext 已经是 16kHz ----
    if (!this.needsResample) {
      const int16 = this._floatToInt16(channelData);
      this.port.postMessage(int16.buffer, [int16.buffer]);
      return true;
    }

    // ---- 降采样路径：从 inputRate → 16kHz ----

    // 1. 将新样本追加到输入缓冲区
    const needed = this.inputBufLen + channelData.length;
    if (needed > this.inputBuf.length) {
      // 扩容：至少翻倍或满足当前需求
      const newSize = Math.max(needed, this.inputBuf.length * 2);
      const newBuf = new Float32Array(newSize);
      newBuf.set(this.inputBuf.subarray(0, this.inputBufLen), 0);
      this.inputBuf = newBuf;
    }
    this.inputBuf.set(channelData, this.inputBufLen);
    this.inputBufLen += channelData.length;

    // 2. 计算可以产出多少 16kHz 输出样本
    // ratio = inputRate / TARGET_RATE（每个输出样本需要多少输入样本）
    // 例如 44100/16000 ≈ 2.7625, 48000/16000 = 3.0
    const ratio = this.inputRate / TARGET_RATE;
    const availableInput = this.inputBufLen - this.resamplePos;
    const outputCount = Math.floor(availableInput / ratio);

    if (outputCount < 1) return true; // 输入不够，等下次

    // 3. 线性插值降采样
    const output = new Float32Array(outputCount);
    for (let i = 0; i < outputCount; i++) {
      const srcPos = this.resamplePos + i * ratio;
      const floorIdx = Math.floor(srcPos);
      const ceilIdx = Math.min(floorIdx + 1, this.inputBufLen - 1);
      const frac = srcPos - floorIdx;
      output[i] = this.inputBuf[floorIdx] * (1 - frac) + this.inputBuf[ceilIdx] * frac;
    }

    // 4. 推进降采样位置，裁剪已消耗的输入
    this.resamplePos += outputCount * ratio;
    const consumed = Math.floor(this.resamplePos);
    if (consumed > 0) {
      this.inputBuf.copyWithin(0, consumed, this.inputBufLen);
      this.inputBufLen -= consumed;
      this.resamplePos -= consumed;
    }

    // 5. 转换为 Int16 并发送
    const int16 = this._floatToInt16(output);
    this.port.postMessage(int16.buffer, [int16.buffer]);

    return true; // 保持处理器存活
  }
}

registerProcessor('pcm-processor', PCMProcessor);

/**
 * ゆっくりボイス変換 - Offscreen Document
 *
 * 役割：タブ音声のキャプチャ、バックエンドへの送信、変換結果の再生。
 */

// RVC フル推論用に 1 秒塊（レイテンシ許容）
const CHUNK_SAMPLES = 16000; // 16kHz * 1s
const CAPTURE_SILENCE_GAIN = 0.0001;

let audioContext = null;
let captureStream = null;
let captureSource = null;
let workletNode = null;
let playbackGain = null;
let playbackTime = 0;

let pendingFrames = [];
let pendingSamples = 0;
let uploadQueue = [];
let queueProcessing = false;

const state = {
  active: false,
  sampleRate: 16000,
  backendUrl: '',
};

function notifyBackground(type, payload = {}) {
  chrome.runtime.sendMessage({ from: 'offscreen', type, ...payload }).catch(() => undefined);
}

async function ensureAudioContext(sampleRate) {
  if (audioContext) {
    return audioContext;
  }

  audioContext = new AudioContext({ sampleRate });
  await audioContext.audioWorklet.addModule('audio-worklet-processor.js');

  playbackGain = audioContext.createGain();
  playbackGain.gain.setValueAtTime(1.0, audioContext.currentTime);
  playbackGain.connect(audioContext.destination);

  return audioContext;
}

async function startPipeline({ streamId, backendUrl, sampleRate }) {
  if (state.active) {
    return { ok: true };
  }

  try {
  // 強制的にフル変換エンドポイントへ差し替え
  state.backendUrl = backendUrl.replace(/\/process_chunk|\/convert_full|\/convert$/, '/convert_full');
    state.sampleRate = sampleRate;

    const constraints = {
      audio: {
        mandatory: {
          chromeMediaSource: 'tab',
          chromeMediaSourceId: streamId,
        },
      },
      video: false,
    };

    captureStream = await navigator.mediaDevices.getUserMedia(constraints);
    const ctx = await ensureAudioContext(sampleRate);
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }

    captureSource = ctx.createMediaStreamSource(captureStream);
    workletNode = new AudioWorkletNode(ctx, 'capture-processor');

    workletNode.port.onmessage = (event) => {
      if (event.data instanceof Float32Array) {
        handleAudioFrame(event.data);
      }
    };

    const silenceGain = ctx.createGain();
    silenceGain.gain.setValueAtTime(CAPTURE_SILENCE_GAIN, ctx.currentTime);

    captureSource.connect(workletNode);
    workletNode.connect(silenceGain);
    silenceGain.connect(ctx.destination);

    state.active = true;
    playbackTime = ctx.currentTime;
    pendingFrames = [];
    pendingSamples = 0;
    uploadQueue = [];
    queueProcessing = false;

    notifyBackground('OFFSCREEN_STARTED');
    return { ok: true };
  } catch (error) {
    notifyBackground('OFFSCREEN_ERROR', { error: error?.message || String(error) });
    await stopPipeline();
    return { ok: false, error: error?.message || String(error) };
  }
}

async function stopPipeline() {
  if (!state.active) {
    return { ok: true };
  }

  try {
    if (workletNode) {
      workletNode.port.onmessage = null;
      workletNode.disconnect();
    }
    if (captureSource) {
      captureSource.disconnect();
    }
    if (captureStream) {
      captureStream.getTracks().forEach((track) => track.stop());
    }
    if (audioContext) {
      await audioContext.close();
    }
  } catch (error) {
    notifyBackground('OFFSCREEN_ERROR', { error: error?.message || String(error) });
  } finally {
    audioContext = null;
    captureStream = null;
    captureSource = null;
    workletNode = null;
    playbackGain = null;
    playbackTime = 0;
    pendingFrames = [];
    pendingSamples = 0;
    uploadQueue = [];
    queueProcessing = false;
    state.active = false;
    notifyBackground('OFFSCREEN_STOPPED');
  }

  return { ok: true };
}

function handleAudioFrame(frame) {
  if (!state.active || !(frame instanceof Float32Array)) {
    return;
  }

  pendingFrames.push({ data: frame, index: 0 });
  pendingSamples += frame.length;
  flushPendingFrames();
}

function flushPendingFrames() {
  while (pendingSamples >= CHUNK_SAMPLES) {
    const chunk = new Float32Array(CHUNK_SAMPLES);
    let filled = 0;

    while (filled < CHUNK_SAMPLES && pendingFrames.length > 0) {
      const head = pendingFrames[0];
      const available = head.data.length - head.index;
      const needed = CHUNK_SAMPLES - filled;
      const copyCount = Math.min(available, needed);

      const slice = head.data.subarray(head.index, head.index + copyCount);
      chunk.set(slice, filled);

      head.index += copyCount;
      filled += copyCount;

      if (head.index >= head.data.length) {
        pendingFrames.shift();
      }
    }

    pendingSamples -= CHUNK_SAMPLES;
    enqueueChunk(chunk);
  }
}

function enqueueChunk(chunk) {
  uploadQueue.push(chunk);
  processUploadQueue();
}

async function processUploadQueue() {
  if (queueProcessing) {
    return;
  }
  queueProcessing = true;

  while (uploadQueue.length > 0 && state.active) {
    const chunk = uploadQueue.shift();
    try {
      await transmitChunk(chunk);
    } catch (error) {
      notifyBackground('OFFSCREEN_ERROR', { error: error?.message || String(error) });
    }
  }

  queueProcessing = false;
}

async function transmitChunk(chunk) {
  const ctx = audioContext;
  if (!ctx || !state.active) {
    return;
  }

  const payload = chunk.buffer.slice(0);
  const startedAt = performance.now();

  // 強制的に RVC ターゲット (zundamon-1) を付与
  const url = state.backendUrl.includes('?') ? `${state.backendUrl}&target=zundamon-1` : `${state.backendUrl}?target=zundamon-1`;
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/octet-stream',
    },
    body: payload,
  });

  if (!response.ok) {
    throw new Error(`バックエンドエラー: HTTP ${response.status}`);
  }

  const resultBuffer = await response.arrayBuffer();
  schedulePlayback(resultBuffer);

  const latency = performance.now() - startedAt;
  notifyBackground('OFFSCREEN_CHUNK_PROCESSED', { latencyMs: latency });
}

function schedulePlayback(int16Buffer) {
  if (!audioContext || !playbackGain) {
    return;
  }

  if (!int16Buffer || int16Buffer.byteLength === 0) {
    return;
  }

  const int16Array = new Int16Array(int16Buffer);
  if (int16Array.length === 0) {
    return;
  }

  const float32 = new Float32Array(int16Array.length);
  for (let i = 0; i < int16Array.length; i++) {
    float32[i] = Math.max(-1, Math.min(1, int16Array[i] / 32768));
  }

  const buffer = audioContext.createBuffer(1, float32.length, state.sampleRate);
  buffer.copyToChannel(float32, 0);

  const source = audioContext.createBufferSource();
  source.buffer = buffer;
  source.connect(playbackGain);

  if (playbackTime < audioContext.currentTime) {
    playbackTime = audioContext.currentTime;
  }

  source.start(playbackTime);
  playbackTime += buffer.duration;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.target !== 'offscreen') {
    return false;
  }

  if (message.type === 'OFFSCREEN_START') {
    startPipeline(message)
      .then(sendResponse)
      .catch((error) => sendResponse({ ok: false, error: error?.message || String(error) }));
    return true;
  }

  if (message.type === 'OFFSCREEN_STOP') {
    stopPipeline()
      .then(sendResponse)
      .catch((error) => sendResponse({ ok: false, error: error?.message || String(error) }));
    return true;
  }

  return false;
});

console.log('[offscreen] Ready');

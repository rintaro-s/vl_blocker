/**
 * ゆっくりボイス変換 - Background Service Worker
 *
 * 役割:
 *  - UI (popup) からの指示を受けてセッションを管理
 *  - Offscreen ドキュメントと通信し、音声キャプチャ/再生パイプラインを制御
 *  - バックエンドの状況を監視
 */

const BACKEND_BASE_URL = 'http://127.0.0.1:5000';
const HEALTH_ENDPOINT = `${BACKEND_BASE_URL}/health`;
// フル変換エンドポイントへ直接誘導
const PROCESS_ENDPOINT = `${BACKEND_BASE_URL}/convert_full`;

const session = {
  active: false,
  tabId: null,
  streamId: null,
  processedChunks: 0,
  totalLatencyMs: 0,
  lastError: null,
};

async function ensureOffscreenDocument() {
  const existing = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
  });

  if (existing.length > 0) {
    return;
  }

  await chrome.offscreen.createDocument({
    url: 'offscreen.html',
    reasons: ['USER_MEDIA', 'AUDIO_PLAYBACK'],
    justification: 'Process and play converted audio',
  });
}

async function ensureBackendAlive() {
  const response = await fetch(HEALTH_ENDPOINT, { method: 'GET' });
  if (!response.ok) {
    throw new Error('バックエンドが応答しません');
  }
}

function getMediaStreamId(tabId) {
  return new Promise((resolve, reject) => {
    chrome.tabCapture.getMediaStreamId(
      { targetTabId: tabId },
      (streamId) => {
        const err = chrome.runtime.lastError;
        if (err) {
          reject(new Error(err.message));
          return;
        }
        if (!streamId) {
          reject(new Error('タブのオーディオを取得できません')); 
          return;
        }
        resolve(streamId);
      }
    );
  });
}

function sendToTab(tabId, message) {
  return new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(tabId, message, (response) => {
      const err = chrome.runtime.lastError;
      if (err) {
        reject(new Error(err.message));
        return;
      }
      resolve(response);
    });
  });
}

function sendToOffscreen(message) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(
      { target: 'offscreen', ...message },
      (response) => {
        const err = chrome.runtime.lastError;
        if (err) {
          reject(new Error(err.message));
          return;
        }
        resolve(response);
      }
    );
  });
}

function resetBadge() {
  chrome.action.setBadgeText({ text: '' });
}

function setActiveBadge() {
  chrome.action.setBadgeText({ text: 'ON' });
  chrome.action.setBadgeBackgroundColor({ color: '#4CAF50' });
}

async function startConversion(tabId) {
  if (session.active) {
    return { ok: true };
  }

  try {
    await ensureBackendAlive();
    await ensureOffscreenDocument();

    const streamId = await getMediaStreamId(tabId);
    await sendToTab(tabId, { type: 'MUTE_PAGE_AUDIO' }).catch(() => undefined);

    session.active = true;
    session.tabId = tabId;
    session.streamId = streamId;
    session.processedChunks = 0;
    session.totalLatencyMs = 0;
    session.lastError = null;

    await sendToOffscreen({
      type: 'OFFSCREEN_START',
      streamId,
      backendUrl: PROCESS_ENDPOINT,
      sampleRate: 16000,
    });

    setActiveBadge();
    return { ok: true };
  } catch (error) {
    session.active = false;
    session.tabId = null;
    session.streamId = null;
    session.lastError = error.message;
    resetBadge();
    return { ok: false, error: error.message };
  }
}

async function stopConversion() {
  if (!session.active) {
    return { ok: true };
  }

  try {
    await sendToOffscreen({ type: 'OFFSCREEN_STOP' }).catch(() => undefined);
    if (session.tabId !== null) {
      await sendToTab(session.tabId, { type: 'UNMUTE_PAGE_AUDIO' }).catch(() => undefined);
    }
  } finally {
    session.active = false;
    session.tabId = null;
    session.streamId = null;
    resetBadge();
  }

  return { ok: true };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.from === 'offscreen') {
    handleOffscreenMessage(message);
    return false;
  }

  if (message.type === 'START_CONVERSION') {
    const tabId = message.tabId || (sender.tab && sender.tab.id);
    if (!tabId) {
      sendResponse({ ok: false, error: 'タブが見つかりません' });
      return false;
    }

    startConversion(tabId).then(sendResponse);
    return true;
  }

  if (message.type === 'STOP_CONVERSION') {
    stopConversion().then(sendResponse);
    return true;
  }

  if (message.type === 'CONVERSION_STATUS') {
    const avgLatency = session.processedChunks > 0
      ? session.totalLatencyMs / session.processedChunks
      : 0;

    sendResponse({
      ok: true,
      isConverting: session.active,
      metrics: {
        processedChunks: session.processedChunks,
        averageLatencyMs: avgLatency,
      },
      lastError: session.lastError,
    });
    return false;
  }

  return false;
});

function handleOffscreenMessage(message) {
  switch (message.type) {
    case 'OFFSCREEN_STARTED':
      console.log('[background] Offscreen pipeline started');
      break;
    case 'OFFSCREEN_CHUNK_PROCESSED':
      session.processedChunks += 1;
      session.totalLatencyMs += message.latencyMs || 0;
      break;
    case 'OFFSCREEN_ERROR':
      session.lastError = message.error || '不明なエラー';
      console.error('[background] Offscreen error:', message.error);
      break;
    case 'OFFSCREEN_STOPPED':
      console.log('[background] Offscreen pipeline stopped');
      break;
    default:
      break;
  }
}

chrome.action.onClicked.addListener(async (tab) => {
  if (session.active) {
    await stopConversion();
  } else if (tab?.id != null) {
    await startConversion(tab.id);
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  if (session.active && session.tabId === tabId) {
    stopConversion();
  }
});

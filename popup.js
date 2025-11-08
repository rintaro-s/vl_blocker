/**
 * ゆっくりボイス変換 - ポップアップUI
 */

const CONFIG = {
  BACKEND_URL: 'http://localhost:5000'
};

// UI要素
const elements = {
  toggleButton: document.getElementById('toggleButton'),
  statusIndicator: document.getElementById('statusIndicator'),
  detectedSpeaker: document.getElementById('detectedSpeaker'),
  latencyValue: document.getElementById('latencyValue'),
  queueValue: document.getElementById('queueValue'),
  targetVoice: document.getElementById('targetVoice'),
  errorMessage: document.getElementById('errorMessage')
};

// Deprecated duplicate popup script. Use extension/popup.js
console.warn('Deprecated root popup.js');

let isConverting = false;
let metricsInterval = null;

/**
 * 初期化
 */
async function initialize() {
  // バックエンド接続確認
  await checkBackendStatus();
  
  // 現在の状態を取得
  chrome.storage.local.get(['isConverting', 'targetVoice'], (result) => {
    isConverting = result.isConverting || false;
    updateUI();
    
    if (result.targetVoice) {
      elements.targetVoice.value = result.targetVoice;
    }
  });
  
  // イベントリスナー設定
  elements.toggleButton.addEventListener('click', toggleConversion);
  elements.targetVoice.addEventListener('change', updateTargetVoice);
  
  // メトリクス更新開始
  startMetricsUpdate();
}

/**
 * バックエンド接続確認
 */
async function checkBackendStatus() {
  try {
    const response = await fetch(`${CONFIG.BACKEND_URL}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(3000)
    });
    
    if (response.ok) {
      const data = await response.json();
      elements.backendStatus.textContent = '✓ 接続中';
      elements.backendStatus.style.color = '#4CAF50';
      
      // モデル情報を表示
      if (data.models_loaded && data.models_loaded.length > 0) {
        console.log('Loaded models:', data.models_loaded);
      }
      
      return true;
    } else {
      throw new Error('Backend returned error');
    }
    
  } catch (error) {
    elements.backendStatus.textContent = '✗ 未接続';
    elements.backendStatus.style.color = '#f44336';
    showError('バックエンドに接続できません。localhost:5000でサーバーが起動しているか確認してください。');
    return false;
  }
}

/**
 * 変換のトグル
 */
async function toggleConversion() {
  if (isConverting) {
    await stopConversion();
  } else {
    await startConversion();
  }
}

/**
 * 変換開始
 */
async function startConversion() {
  // バックエンド確認
  const backendOk = await checkBackendStatus();
  if (!backendOk) {
    return;
  }
  
  try {
    // アクティブなタブを取得
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab) {
      throw new Error('アクティブなタブが見つかりません');
    }
    
    // バックグラウンドに開始指示
    chrome.runtime.sendMessage({ type: 'START_CONVERSION', tabId: tab.id });
    
    // 状態更新
    isConverting = true;
    chrome.storage.local.set({ isConverting: true });
    updateUI();
    
  } catch (error) {
    console.error('Failed to start conversion:', error);
    showError(`変換開始に失敗しました: ${error.message}`);
  }
}

/**
 * 変換停止
 */
async function stopConversion() {
  try {
    // バックグラウンドに停止指示
    chrome.runtime.sendMessage({ type: 'STOP_CONVERSION' });
    
    // 状態更新
    isConverting = false;
    chrome.storage.local.set({ isConverting: false });
    updateUI();
    
  } catch (error) {
    console.error('Failed to stop conversion:', error);
    showError(`変換停止に失敗しました: ${error.message}`);
  }
}

/**
 * ターゲット音声の更新
 */
function updateTargetVoice() {
  const targetVoice = elements.targetVoice.value;
  chrome.storage.local.set({ targetVoice });
  
  // バックグラウンドに通知
  chrome.runtime.sendMessage({
    type: 'UPDATE_TARGET_VOICE',
    targetVoice
  });
}

/**
 * UI更新
 */
function updateUI() {
  if (isConverting) {
    elements.toggleButton.textContent = '変換を停止';
    elements.toggleButton.classList.remove('start');
    elements.toggleButton.classList.add('stop');
    
    elements.statusIndicator.classList.remove('inactive');
    elements.statusIndicator.classList.add('active');
    elements.statusText.textContent = '変換中';
    
    elements.targetVoice.disabled = true;
    
  } else {
    elements.toggleButton.textContent = '変換を開始';
    elements.toggleButton.classList.remove('stop');
    elements.toggleButton.classList.add('start');
    
    elements.statusIndicator.classList.remove('active');
    elements.statusIndicator.classList.add('inactive');
    elements.statusText.textContent = '停止中';
    
    elements.targetVoice.disabled = false;
    
    // メトリクスをリセット
    elements.latencyValue.textContent = '-';
    elements.queueValue.textContent = '-';
    elements.detectedSpeaker.textContent = '-';
  }
}

/**
 * メトリクス更新開始
 */
function startMetricsUpdate() {
  metricsInterval = setInterval(async () => {
    if (!isConverting) return;
    
    try {
      const response = await fetch(`${CONFIG.BACKEND_URL}/metrics`, {
        signal: AbortSignal.timeout(2000)
      });
      
      if (response.ok) {
        const metrics = await response.json();
        
        if (metrics.latency_p50) {
          elements.latencyValue.textContent = Math.round(metrics.latency_p50);
        }
        
        // その他のメトリクス表示
        // (キューサイズはoffscreenから取得する必要がある)
      }
    } catch (error) {
      // エラーは無視（バックエンドが応答しない場合など）
    }
  }, 1000);
}

/**
 * エラーメッセージ表示
 */
function showError(message) {
  elements.errorMessage.textContent = message;
  elements.errorMessage.classList.add('show');
  
  setTimeout(() => {
    elements.errorMessage.classList.remove('show');
  }, 5000);
}

/**
 * バックグラウンドからのメッセージ処理
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'METRICS_UPDATE') {
    if (message.latency) {
      elements.latencyValue.textContent = Math.round(message.latency);
    }
    if (message.queueSize !== undefined) {
      elements.queueValue.textContent = message.queueSize;
    }
    if (message.speaker) {
      elements.detectedSpeaker.textContent = message.speaker;
    }
  }
  
  else if (message.type === 'CONVERSION_STOPPED') {
    isConverting = false;
    updateUI();
  }
  
  else if (message.type === 'ERROR') {
    showError(message.error);
  }
  
  return true;
});

/**
 * ポップアップが閉じられる前のクリーンアップ
 */
window.addEventListener('beforeunload', () => {
  if (metricsInterval) {
    clearInterval(metricsInterval);
  }
});

// 初期化実行
initialize();
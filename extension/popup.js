/**
 * ゆっくりボイス変換 - Popup UI
 */

const CONFIG = {
  BACKEND_URL: 'http://127.0.0.1:5000',
};

const UI = {
  toggleButton: document.getElementById('toggleButton'),
  statusBox: document.getElementById('statusBox'),
  statusText: document.getElementById('statusText'),
  statusIndicator: document.querySelector('.status-indicator'),
  metricsBox: document.getElementById('metricsBox'),
  processedChunks: document.getElementById('processedChunks'),
  latency: document.getElementById('latency'),
  backendStatus: document.getElementById('backendStatus'),
  backendText: document.getElementById('backendText'),
  backendDot: document.querySelector('.backend-dot'),
  errorMessage: document.getElementById('errorMessage'),
  speakersSection: document.getElementById('speakersSection'),
  speakerList: document.getElementById('speakerList'),
};

let isConverting = false;
let statusCheckInterval = null;

/**
 * 初期化
 */
async function initialize() {
  try {
    // バックエンド確認
    await checkBackendStatus();

    // 話者一覧を取得
    await loadSpeakers();

    // UIイベント
    UI.toggleButton.addEventListener('click', toggleConversion);

    // 定期的に状態を確認
    statusCheckInterval = setInterval(updateStatus, 500);

    // ポップアップを閉じる時にクリーンアップ
    window.addEventListener('beforeunload', () => {
      clearInterval(statusCheckInterval);
    });
  } catch (error) {
    console.error('Initialization error:', error);
    showError('初期化に失敗しました');
  }
}

/**
 * バックエンド状態確認
 */
async function checkBackendStatus() {
  try {
    const response = await fetch(`${CONFIG.BACKEND_URL}/health`, {
      signal: AbortSignal.timeout(3000),
    });

    if (response.ok) {
      const data = await response.json();
      UI.backendStatus.classList.add('ok');
      UI.backendDot.classList.add('ok');
      UI.backendText.textContent = `バックエンド: ${data.device}`;
      return true;
    }
  } catch (error) {
    UI.backendStatus.classList.remove('ok');
    UI.backendDot.classList.remove('ok');
    UI.backendText.textContent = 'バックエンド: 未接続';
    showError('バックエンド (localhost:5000) に接続できません');
    return false;
  }
}

/**
 * 話者一覧を取得
 */
async function loadSpeakers() {
  try {
    const response = await fetch(`${CONFIG.BACKEND_URL}/speakers`);
    if (!response.ok) throw new Error('Failed to load speakers');

    const data = await response.json();

    if (data.speakers && data.speakers.length > 0) {
      UI.speakersSection.style.display = 'block';
      UI.speakerList.innerHTML = '';

      data.speakers.forEach((speaker) => {
        const li = document.createElement('li');
        li.textContent = speaker.name;
        UI.speakerList.appendChild(li);
      });
    }
  } catch (error) {
    console.error('Failed to load speakers:', error);
  }
}

/**
 * 変換をトグル
 */
async function toggleConversion() {
  try {
    const [tab] = await chrome.tabs.query({
      active: true,
      currentWindow: true,
    });

    if (!tab) {
      showError('アクティブなタブが見つかりません');
      return;
    }

    let response;
    if (isConverting) {
      response = await chrome.runtime.sendMessage({ type: 'STOP_CONVERSION', tabId: tab.id });
    } else {
      response = await chrome.runtime.sendMessage({ type: 'START_CONVERSION', tabId: tab.id });
    }

    if (response?.ok === false) {
      showError(response.error || '操作に失敗しました');
    }

    await updateStatus();
    updateUI();
  } catch (error) {
    console.error('Toggle error:', error);
    showError(`操作に失敗しました: ${error.message}`);
  }
}

/**
 * 状態を更新
 */
async function updateStatus() {
  try {
    const response = await chrome.runtime.sendMessage({ type: 'CONVERSION_STATUS' });

    if (response?.ok) {
      isConverting = response.isConverting;
      const metrics = response.metrics || {};

      if (typeof metrics.processedChunks === 'number') {
        UI.processedChunks.textContent = metrics.processedChunks;
      }

      if (typeof metrics.averageLatencyMs === 'number' && metrics.averageLatencyMs > 0) {
        UI.latency.textContent = `${metrics.averageLatencyMs.toFixed(0)}ms`;
      } else {
        UI.latency.textContent = '-';
      }

      if (response.lastError) {
        showError(response.lastError);
      }

      updateUI();
    }
  } catch (error) {
    // 無視（バックグラウンドスクリプトがまだロードされていない可能性）
  }
}

/**
 * UI を更新
 */
function updateUI() {
  if (isConverting) {
    UI.toggleButton.textContent = '⏹ 変換を停止';
    UI.toggleButton.classList.remove('start');
    UI.toggleButton.classList.add('stop');

    UI.statusBox.classList.add('active');
    UI.statusIndicator.classList.add('active');
    UI.statusText.textContent = '変換中';

    UI.metricsBox.style.display = 'block';
  } else {
    UI.toggleButton.textContent = '🎙️ 変換を開始';
    UI.toggleButton.classList.remove('stop');
    UI.toggleButton.classList.add('start');

    UI.statusBox.classList.remove('active');
    UI.statusIndicator.classList.remove('active');
    UI.statusText.textContent = '停止中';

    UI.metricsBox.style.display = 'none';
  }
}

/**
 * エラーメッセージ表示
 */
function showError(message) {
  UI.errorMessage.textContent = message;
  UI.errorMessage.classList.add('show');

  setTimeout(() => {
    UI.errorMessage.classList.remove('show');
  }, 5000);
}

// 初期化実行
initialize();

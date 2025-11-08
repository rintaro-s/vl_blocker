/**
 * ゆっくりボイス変換 - コンテンツスクリプト
 * ページ内の音声要素を制御
 */

let originalVolume = {};
let mutedElements = new Set();

/**
 * タブの音声をミュート
 */
function muteTab() {
  console.log('Muting tab audio...');
  
  // <video> 要素をミュート
  document.querySelectorAll('video').forEach(video => {
    if (!video.muted) {
      originalVolume[video] = video.volume;
      video.muted = true;
      mutedElements.add(video);
    }
  });
  
  // <audio> 要素をミュート
  document.querySelectorAll('audio').forEach(audio => {
    if (!audio.muted) {
      originalVolume[audio] = audio.volume;
      audio.muted = true;
      mutedElements.add(audio);
    }
  });
  
  // Web Audio API のコンテキストをサスペンド（可能な場合）
  // Note: 外部から直接制御できないため、ページ側のコードに依存
  
  console.log(`Muted ${mutedElements.size} elements`);
}

/**
 * タブの音声ミュートを解除
 */
function unmuteTab() {
  console.log('Unmuting tab audio...');
  
  mutedElements.forEach(element => {
    element.muted = false;
    if (originalVolume[element] !== undefined) {
      element.volume = originalVolume[element];
    }
  });
  
  mutedElements.clear();
  originalVolume = {};
  
  console.log('Tab audio unmuted');
}

/**
 * 動的に追加される音声要素を監視
 */
const observer = new MutationObserver((mutations) => {
  if (mutedElements.size > 0) {
    // 変換中の場合、新しく追加された要素もミュート
    mutations.forEach(mutation => {
      mutation.addedNodes.forEach(node => {
        if (node.nodeType === Node.ELEMENT_NODE) {
          if (node.tagName === 'VIDEO' || node.tagName === 'AUDIO') {
            originalVolume[node] = node.volume;
            node.muted = true;
            mutedElements.add(node);
          }
          
          // 子要素もチェック
          node.querySelectorAll && node.querySelectorAll('video, audio').forEach(media => {
            originalVolume[media] = media.volume;
            media.muted = true;
            mutedElements.add(media);
          });
        }
      });
    });
  }
});

// DOM監視開始
observer.observe(document.body, {
  childList: true,
  subtree: true
});

/**
 * バックグラウンドからのメッセージ処理
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'MUTE_TAB') {
    muteTab();
    sendResponse({ status: 'muted' });
  }
  
  else if (message.type === 'UNMUTE_TAB') {
    unmuteTab();
    sendResponse({ status: 'unmuted' });
  }
  
  return true;
});

/**
 * ページアンロード時のクリーンアップ
 */
window.addEventListener('beforeunload', () => {
  unmuteTab();
});

console.log('ゆっくりボイス変換 content script loaded');
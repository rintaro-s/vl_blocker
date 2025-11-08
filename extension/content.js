/**
 * ゆっくりボイス変換 - Content Script
 * 
 * 役割：ページ側の音声をミュート / 復元するだけ。
 * キャプチャと変換処理は offscreen ドキュメントで行う。
 */

console.log('[content] Ready');

const mutedElements = new Set();
const originalVolumes = new Map();

function mutePage() {
  document.querySelectorAll('audio, video').forEach((el) => {
    if (!el.muted) {
      originalVolumes.set(el, el.volume);
      el.muted = true;
      mutedElements.add(el);
    }
  });
}

function unmutePage() {
  mutedElements.forEach((el) => {
    el.muted = false;
    if (originalVolumes.has(el)) {
      el.volume = originalVolumes.get(el);
    }
  });
  mutedElements.clear();
  originalVolumes.clear();
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'MUTE_PAGE_AUDIO') {
    mutePage();
    sendResponse({ ok: true });
    return true;
  }

  if (message.type === 'UNMUTE_PAGE_AUDIO') {
    unmutePage();
    sendResponse({ ok: true });
    return true;
  }

  return false;
});

const observer = new MutationObserver((mutations) => {
  if (mutedElements.size === 0) {
    return;
  }

  mutations.forEach((mutation) => {
    mutation.addedNodes.forEach((node) => {
      if (node.nodeType === Node.ELEMENT_NODE) {
        if (node.tagName === 'AUDIO' || node.tagName === 'VIDEO') {
          node.muted = true;
          if (originalVolumes.has(node)) {
            node.volume = originalVolumes.get(node);
          }
          mutedElements.add(node);
        }
      }
    });
  });
});

observer.observe(document.documentElement, {
  childList: true,
  subtree: true,
});

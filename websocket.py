"""
WebSocketサーバー実装
低レイテンシのリアルタイム音声ストリーミング用
"""

from flask_sock import Sock
import numpy as np
import threading
import queue
import time
from pathlib import Path

# voice_converter.py に追加する内容

# グローバルにSockを追加
# sock = Sock(app)  # will be initialized in voice_converter main

class AudioStreamProcessor:
    """リアルタイム音声ストリーミング処理"""
    
    def __init__(self, converter, buffer_size=3):
        self.converter = converter
        self.buffer_size = buffer_size
        self.processing_queue = queue.Queue(maxsize=buffer_size)
        self.output_queue = queue.Queue(maxsize=buffer_size)
        self.is_running = False
        self.process_thread = None
        
    def start(self):
        """処理スレッド開始"""
        self.is_running = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
    
    def stop(self):
        """処理スレッド停止"""
        self.is_running = False
        if self.process_thread:
            self.process_thread.join(timeout=2)
    
    def _process_loop(self):
        """音声処理ループ"""
        while self.is_running:
            try:
                # キューから音声データ取得 (タイムアウト付き)
                audio_data, target_voice = self.processing_queue.get(timeout=0.1)
                
                # 処理開始時刻
                start_time = time.time()
                
                # 音声変換
                converted = self.converter.process_audio_chunk(audio_data, target_voice)
                
                # 処理時間計測
                process_time = (time.time() - start_time) * 1000  # ms
                
                # 出力キューに追加
                self.output_queue.put({
                    'audio': converted,
                    'latency': process_time
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def add_audio(self, audio_data, target_voice='natural'):
        """音声をキューに追加"""
        try:
            self.processing_queue.put_nowait((audio_data, target_voice))
            return True
        except queue.Full:
            # キューが満杯の場合は古いデータを破棄
            try:
                self.processing_queue.get_nowait()
                self.processing_queue.put_nowait((audio_data, target_voice))
                return True
            except:
                return False
    
    def get_output(self, timeout=0.1):
        """変換済み音声を取得"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# WebSocket handler would be registered in voice_converter.py where the Flask app exists.


class PerformanceMonitor:
    """パフォーマンス監視"""
    
    def __init__(self):
        self.latencies = []
        self.throughput = []
        self.gpu_usage = []
        
    def record_latency(self, latency_ms):
        self.latencies.append(latency_ms)
        # 最新1000件のみ保持
        if len(self.latencies) > 1000:
            self.latencies = self.latencies[-1000:]
    
    def get_stats(self):
        if not self.latencies:
            return {}
        
        import numpy as _np
        return {
            'latency_avg': _np.mean(self.latencies),
            'latency_p50': _np.percentile(self.latencies, 50),
            'latency_p95': _np.percentile(self.latencies, 95),
            'latency_p99': _np.percentile(self.latencies, 99),
            'throughput': len(self.latencies) / 60  # chunks per second
        }


# Helper for integration: initialize Sock with app in voice_converter.py
def initialize_websocket_server(app):
    print("Initializing WebSocket server...")
    sock = Sock(app)
    print("WebSocket server ready at ws://localhost:5000/stream")
    return sock

"""
リアルタイムゆっくりボイス→自然な音声変換システム
GPU対応バックエンド
"""

import torch
import torchaudio
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import librosa
from scipy import signal
import json
import os
from pathlib import Path

# Demucs (音源分離) - Meta製
from demucs.pretrained import get_model
from demucs.apply import apply_model

# RVC (Retrieval-based Voice Conversion) - 軽量で高品質
import sys
sys.path.append('./rvc')
from infer.modules.vc.modules import VC

app = Flask(__name__)
CORS(app)

class VoiceConverter:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Demucs for vocal separation (htdemucs_ft を使用 - 高速版)
        print("Loading Demucs model...")
        self.separator = get_model('htdemucs_ft')
        self.separator.to(self.device)
        self.separator.eval()
        
        # RVC models (ゆっくり→人間音声変換用)
        self.rvc_models = {}
        self.load_rvc_models()
        
        # ゆっくりボイス識別用の特徴量
        self.yukkuri_profiles = self.load_yukkuri_profiles()
        
        # 処理設定
        self.sample_rate = 16000
        self.chunk_size = 16000  # 1秒チャンク
        self.overlap = 4000  # 250ms オーバーラップ
    
    def load_rvc_models(self):
        """RVCモデルをロード"""
        models_dir = Path('./models/rvc')
        if not models_dir.exists():
            print("RVC models directory not found. Creating...")
            models_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # 各ゆっくりキャラクター用のRVCモデルをロード
        for model_file in models_dir.glob('*.pth'):
            character_name = model_file.stem
            print(f"Loading RVC model for {character_name}...")
            try:
                self.rvc_models[character_name] = {
                    'model_path': str(model_file),
                    'index_path': str(model_file.with_suffix('.index')),
                    'vc': VC(self.device)
                }
            except Exception as e:
                print(f"Failed to load {character_name}: {e}")
    
    def load_yukkuri_profiles(self):
        """ゆっくりボイスの音響プロファイルをロード"""
        profiles_path = Path('./profiles/yukkuri_profiles.json')
        if profiles_path.exists():
            with open(profiles_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # デフォルトプロファイル
        return {
            'reimu': {'f0_range': [180, 250], 'formants': [800, 1200, 2500]},
            'marisa': {'f0_range': [150, 220], 'formants': [700, 1100, 2400]},
            'zundamon': {'f0_range': [200, 280], 'formants': [850, 1300, 2600]}
        }
    
    def separate_vocals(self, audio_tensor):
        """
        音声とBGM/効果音を分離
        Returns: (vocals, accompaniment)
        """
        with torch.no_grad():
            # Demucsは4つのstem(vocals, drums, bass, other)を返す
            sources = apply_model(self.separator, audio_tensor.unsqueeze(0).to(self.device))
            vocals = sources[0, 0]  # vocals stem
            # その他を合成
            accompaniment = sources[0, 1:].sum(dim=0)
        
        return vocals.cpu(), accompaniment.cpu()
    
    def identify_yukkuri_speaker(self, vocal_audio):
        """
        ゆっくりボイスの話者を識別
        Returns: character_name or 'unknown'
        """
        # F0 (基本周波数) を抽出
        audio_np = vocal_audio.numpy()
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_np,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        # 有声部分の平均F0
        f0_mean = np.nanmean(f0[voiced_flag])
        
        if np.isnan(f0_mean):
            return 'unknown'
        
        # プロファイルと照合
        best_match = 'unknown'
        min_distance = float('inf')
        
        for char_name, profile in self.yukkuri_profiles.items():
            f0_range = profile['f0_range']
            if f0_range[0] <= f0_mean <= f0_range[1]:
                distance = abs(f0_mean - (f0_range[0] + f0_range[1]) / 2)
                if distance < min_distance:
                    min_distance = distance
                    best_match = char_name
        
        return best_match
    
    def convert_voice(self, vocal_audio, source_character, target_voice='natural'):
        """
        RVCを使用して音声変換
        """
        if source_character not in self.rvc_models:
            print(f"No RVC model for {source_character}, returning original")
            return vocal_audio
        
        model_info = self.rvc_models[source_character]
        vc = model_info['vc']
        
        try:
            # RVCで変換
            audio_np = vocal_audio.numpy()
            
            # RVC inference
            converted = vc.pipeline(
                model=model_info['model_path'],
                audio_input=audio_np,
                pitch_adjust=0,  # ピッチ調整なし
                index_file=model_info['index_path'],
                index_rate=0.75,
                filter_radius=3,
                volume_envelope=1.0,
                protect_voiceless=0.5,
                hop_length=128,
                f0_method='rmvpe'  # 高精度F0抽出
            )
            
            return torch.from_numpy(converted).float()
            
        except Exception as e:
            print(f"Voice conversion failed: {e}")
            return vocal_audio
    
    def process_audio_chunk(self, audio_data, target_voice='natural'):
        """
        音声チャンクをリアルタイム処理
        """
        # バイトデータをTensorに変換
        audio_np = np.frombuffer(audio_data, dtype=np.float32)
        audio_tensor = torch.from_numpy(audio_np)
        
        # 1. 音声とBGMを分離
        vocals, accompaniment = self.separate_vocals(audio_tensor)
        
        # 2. ゆっくり話者を識別
        speaker = self.identify_yukkuri_speaker(vocals)
        print(f"Detected speaker: {speaker}")
        
        # 3. 音声変換
        if speaker != 'unknown':
            converted_vocals = self.convert_voice(vocals, speaker, target_voice)
        else:
            converted_vocals = vocals
        
        # 4. BGMと再合成
        output = converted_vocals + accompaniment
        
        # 正規化
        max_val = torch.abs(output).max()
        if max_val > 1.0:
            output = output / max_val
        
        return output.numpy().astype(np.float32).tobytes()

# グローバルインスタンス
converter = VoiceConverter()

@app.route('/health', methods=['GET'])
def health_check():
    """ヘルスチェック"""
    return jsonify({
        'status': 'ok',
        'device': str(converter.device),
        'models_loaded': list(converter.rvc_models.keys())
    })

@app.route('/convert', methods=['POST'])
def convert_audio():
    """
    音声変換エンドポイント
    入力: raw audio data (float32, mono, 16kHz)
    出力: 変換後の音声
    """
    try:
        audio_data = request.data
        target_voice = request.args.get('target', 'natural')
        
        # 処理
        converted_audio = converter.process_audio_chunk(audio_data, target_voice)
        
        return send_file(
            io.BytesIO(converted_audio),
            mimetype='audio/raw',
            as_attachment=False
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/profiles', methods=['GET'])
def get_profiles():
    """ゆっくりプロファイル一覧を取得"""
    return jsonify(converter.yukkuri_profiles)

@app.route('/profiles/<character>', methods=['POST'])
def update_profile(character):
    """プロファイルを更新（学習データから）"""
    try:
        data = request.json
        converter.yukkuri_profiles[character] = data
        
        # 保存
        profiles_path = Path('./profiles/yukkuri_profiles.json')
        profiles_path.parent.mkdir(exist_ok=True)
        with open(profiles_path, 'w', encoding='utf-8') as f:
            json.dump(converter.yukkuri_profiles, f, ensure_ascii=False, indent=2)
        
        return jsonify({'status': 'updated'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """
    新しいゆっくりキャラクター用のRVCモデルをトレーニング
    入力: 音声サンプル (複数) + キャラクター名
    """
    try:
        character = request.form.get('character')
        audio_files = request.files.getlist('audio')
        
        if not character or not audio_files:
            return jsonify({'error': 'character and audio files required'}), 400
        
        # トレーニングディレクトリ作成
        train_dir = Path(f'./training/{character}')
        train_dir.mkdir(parents=True, exist_ok=True)
        
        # 音声ファイルを保存
        for i, audio_file in enumerate(audio_files):
            audio_file.save(train_dir / f'sample_{i}.wav')
        
        # RVCトレーニングスクリプトを呼び出す
        # (実際のトレーニングは別プロセスで実行推奨)
        import subprocess
        result = subprocess.run([
            'python', 'train_rvc.py',
            '--character', character,
            '--data_dir', str(train_dir)
        ], capture_output=True)
        
        if result.returncode == 0:
            # モデルを再読み込み
            converter.load_rvc_models()
            return jsonify({'status': 'training_completed'})
        else:
            return jsonify({'error': result.stderr.decode()}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # ディレクトリ作成
    Path('./models/rvc').mkdir(parents=True, exist_ok=True)
    Path('./profiles').mkdir(parents=True, exist_ok=True)
    Path('./training').mkdir(parents=True, exist_ok=True)
    
    # サーバー起動
    app.run(host='0.0.0.0', port=5000, threaded=True)

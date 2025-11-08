"""
RVCモデルトレーニングスクリプト
ゆっくりボイス用の音声変換モデルを学習
"""

import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path
import json
import librosa
from tqdm import tqdm
import sys

# RVC imports
sys.path.append('./rvc')
from infer.lib.train import utils
from infer.lib.train.data_utils import TextAudioSpeakerLoader
from infer.lib.train.process_ckpt import savee

class YukkuriRVCTrainer:
    def __init__(self, character_name, data_dir, output_dir='./models/rvc'):
        self.character = character_name
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on: {self.device}")
        
        self.sample_rate = 16000
        
    def preprocess_audio(self):
        """音声ファイルの前処理"""
        print("Preprocessing audio files...")
        
        processed_dir = self.data_dir / 'processed'
        processed_dir.mkdir(exist_ok=True)
        
        audio_files = list(self.data_dir.glob('*.wav')) + list(self.data_dir.glob('*.mp3'))
        
        if not audio_files:
            raise ValueError(f"No audio files found in {self.data_dir}")
        
        processed_files = []
        
        for audio_file in tqdm(audio_files):
            try:
                # 音声読み込み
                audio, sr = librosa.load(audio_file, sr=self.sample_rate, mono=True)
                
                # 無音部分をトリミング
                audio, _ = librosa.effects.trim(audio, top_db=20)
                
                # 正規化
                audio = audio / np.max(np.abs(audio))
                
                # 保存
                output_file = processed_dir / f"{audio_file.stem}_processed.wav"
                torchaudio.save(
                    output_file,
                    torch.from_numpy(audio).unsqueeze(0),
                    self.sample_rate
                )
                
                processed_files.append(output_file)
                
            except Exception as e:
                print(f"Failed to process {audio_file}: {e}")
        
        print(f"Processed {len(processed_files)} files")
        return processed_files
    
    def extract_features(self, audio_files):
        """F0とスペクトル特徴を抽出"""
        print("Extracting features...")
        
        features_dir = self.data_dir / 'features'
        features_dir.mkdir(exist_ok=True)
        
        all_f0 = []
        all_formants = []
        
        for audio_file in tqdm(audio_files):
            audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # F0抽出 (RMVPE - RVCの推奨手法)
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            
            # フォルマント抽出
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            # 有効なF0値のみ保存
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                all_f0.extend(valid_f0)
            
            all_formants.extend(spectral_centroids)
            
            # 特徴量を保存
            features = {
                'f0': f0.tolist(),
                'voiced_flag': voiced_flag.tolist()
            }
            
            feature_file = features_dir / f"{audio_file.stem}.json"
            with open(feature_file, 'w') as f:
                json.dump(features, f)
        
        # プロファイル生成
        profile = {
            'f0_range': [float(np.percentile(all_f0, 5)), float(np.percentile(all_f0, 95))],
            'f0_mean': float(np.mean(all_f0)),
            'formants': [
                float(np.percentile(all_formants, 25)),
                float(np.percentile(all_formants, 50)),
                float(np.percentile(all_formants, 75))
            ]
        }
        
        # プロファイル保存
        profile_path = Path('./profiles/yukkuri_profiles.json')
        profile_path.parent.mkdir(exist_ok=True)
        
        if profile_path.exists():
            with open(profile_path, 'r', encoding='utf-8') as f:
                profiles = json.load(f)
        else:
            profiles = {}
        
        profiles[self.character] = profile
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
        
        print(f"Profile saved: {profile}")
        return profile
    
    def train_model(self, epochs=100, batch_size=8):
        """RVCモデルのトレーニング"""
        print(f"Training RVC model for {self.character}...")
        
        # RVCトレーニング設定
        config = {
            'train': {
                'log_interval': 100,
                'eval_interval': 500,
                'seed': 1234,
                'epochs': epochs,
                'learning_rate': 2e-4,
                'betas': [0.8, 0.99],
                'eps': 1e-9,
                'batch_size': batch_size,
                'fp16_run': True,
                'lr_decay': 0.999875,
                'segment_size': 12800,
                'init_lr_ratio': 1,
                'warmup_epochs': 0,
                'c_mel': 45,
                'c_kl': 1.0
            },
            'data': {
                'training_files': str(self.data_dir / 'processed'),
                'segment_size': 12800,
                'filter_length': 2048,
                'hop_length': 320,
                'win_length': 1024,
                'sampling_rate': self.sample_rate,
                'mel_fmin': 0.0,
                'mel_fmax': None
            },
            'model': {
                'inter_channels': 192,
                'hidden_channels': 192,
                'filter_channels': 768,
                'n_heads': 2,
                'n_layers': 6,
                'kernel_size': 3,
                'p_dropout': 0.1,
                'resblock': '1',
                'resblock_kernel_sizes': [3, 7, 11],
                'resblock_dilation_sizes': [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                'upsample_rates': [10, 8, 2, 2],
                'upsample_initial_channel': 512,
                'upsample_kernel_sizes': [20, 16, 4, 4],
                'use_spectral_norm': False,
                'gin_channels': 256,
                'spk_embed_dim': 109
            }
        }
        
        # 設定ファイル保存
        config_path = self.output_dir / f'{self.character}_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # RVC CLIトレーニング実行
        import subprocess
        
        cmd = [
            'python', './rvc/train_nsf_sim_cache_sid_load_pretrain.py',
            '--config', str(config_path),
            '--model', str(self.output_dir / self.character),
            '--data_dir', str(self.data_dir / 'processed')
        ]
        
        print("Starting training process...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Training completed successfully!")
            
            # インデックスファイル作成
            self.create_index()
            
            return True
        else:
            print(f"Training failed: {result.stderr}")
            return False
    
    def create_index(self):
        """検索用インデックスファイルを作成"""
        print("Creating index file...")
        
        try:
            from infer.lib.train.data_utils import build_index
            
            model_path = self.output_dir / f'{self.character}.pth'
            index_path = self.output_dir / f'{self.character}.index'
            
            # インデックス構築
            processed_dir = self.data_dir / 'processed'
            audio_files = list(processed_dir.glob('*.wav'))
            
            # Faiss index作成
            import faiss
            
            # 特徴量抽出
            features_list = []
            for audio_file in tqdm(audio_files):
                audio, sr = librosa.load(audio_file, sr=self.sample_rate)
                
                # メルスペクトログラム
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                
                # 平均プーリング
                features = np.mean(mel_spec_db, axis=1)
                features_list.append(features)
            
            # インデックス構築
            features_array = np.array(features_list).astype('float32')
            
            # L2正規化
            faiss.normalize_L2(features_array)
            
            # インデックス作成
            index = faiss.IndexFlatIP(features_array.shape[1])
            index.add(features_array)
            
            # 保存
            faiss.write_index(index, str(index_path))
            
            print(f"Index saved to {index_path}")
            
        except Exception as e:
            print(f"Failed to create index: {e}")
            print("Index creation is optional, model can still be used.")


def main():
    parser = argparse.ArgumentParser(description='Train RVC model for Yukkuri voice')
    parser.add_argument('--character', type=str, required=True, help='Character name (e.g., reimu, marisa, zundamon)')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training audio files')
    parser.add_argument('--output_dir', type=str, default='./models/rvc', help='Output directory for trained models')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    
    args = parser.parse_args()
    
    # トレーナー初期化
    trainer = YukkuriRVCTrainer(
        character_name=args.character,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # 1. 前処理
    processed_files = trainer.preprocess_audio()
    
    # 2. 特徴量抽出
    profile = trainer.extract_features(processed_files)
    
    # 3. モデルトレーニング
    success = trainer.train_model(epochs=args.epochs, batch_size=args.batch_size)
    
    if success:
        print(f"\n✅ Training completed for {args.character}!")
        print(f"Model saved to: {trainer.output_dir / args.character}.pth")
        print(f"Profile: {profile}")
    else:
        print(f"\n❌ Training failed for {args.character}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

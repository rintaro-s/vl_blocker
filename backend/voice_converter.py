"""
ゆっくりボイス→自然音声変換システム（バックエンド）
Demucs（音源分離）+ RVC（音声変換）を使用したリアルタイム処理
"""

import numpy as np
import soundfile as sf
from scipy import signal
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
import io
import argparse
import sys
import json
from datetime import datetime
import logging
from typing import List, Tuple, Dict, Optional
from threading import Thread
import time

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports (graceful degradation)
HAS_TORCH = False
HAS_LIBROSA = False
HAS_DEMUCS = False

try:
    import torch
    HAS_TORCH = True
    logger.info("torch available")
except ImportError:
    logger.warning("torch not available; running in DSP-only mode")

try:
    import librosa
    HAS_LIBROSA = True
    logger.info("librosa available")
except ImportError:
    logger.warning("librosa not available; F0 detection disabled")

try:
    if HAS_TORCH:
        from demucs.pretrained import get_model
        HAS_DEMUCS = True
        logger.info("demucs available")
except ImportError:
    logger.warning("demucs not available")

app = Flask(__name__)
CORS(app)

# グローバル設定
CONFIG = {
    'SAMPLE_RATE': 16000,
    'CHUNK_SIZE': 16000,  # 1秒
    'DEVICE': 'cuda' if (HAS_TORCH and torch.cuda.is_available()) else 'cpu',
}


class VoiceMappingManager:
    """合成音声ID→ターゲット音声IDのマッピング管理"""

    def __init__(self):
        self.config_dir = Path(__file__).parent / 'config'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.map_path = self.config_dir / 'voice_map.json'
        self.mapping = self._load()

    def _default(self):
        # 既定の対応（存在しないターゲットは後でユーザーが置き換え）
        return {
            "reimu": "natural_female",
            "marisa": "natural_male",
            "zundamon": "natural_boy",
        }

    def _load(self):
        if self.map_path.exists():
            try:
                with open(self.map_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                logger.warning(f"Failed to load voice_map.json: {e}")
        # 初期化
        data = self._default()
        self._save(data)
        return data

    def _save(self, data):
        try:
            with open(self.map_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save voice_map.json: {e}")

    def get_mapping(self):
        return dict(self.mapping)

    def get_target_for(self, synthetic_id: str):
        return self.mapping.get(synthetic_id)

    def validate_update(self, updates: dict, allowed_sources: List[str]) -> dict:
        """validate and clean updates: only allow known source keys and non-empty targets"""
        cleaned = {}
        for k, v in updates.items():
            if not isinstance(k, str):
                continue
            if k not in allowed_sources:
                logger.warning(f"Ignoring unknown source key: {k}")
                continue
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == '':
                continue
            cleaned[k] = v
        return cleaned

    def update(self, updates: dict):
        self.mapping.update(updates)
        self._save(self.mapping)


class VoiceConverter:
    """ゆっくりボイス変換エンジン"""
    
    def __init__(self):
        self.device = torch.device(CONFIG['DEVICE']) if HAS_TORCH else None
        self.sample_rate = CONFIG['SAMPLE_RATE']
        
        # Demucsモデル（音源分離）
        try:
            if HAS_DEMUCS:
                logger.info("Loading Demucs model...")
                self.separator = get_model('htdemucs_ft')
                self.separator.to(self.device)
                self.separator.eval()
                logger.info("✓ Demucs model loaded")
            else:
                self.separator = None
        except Exception as e:
            logger.error(f"Failed to load Demucs: {e}")
            self.separator = None
        
        # RVCモデル（音声変換）
        self.rvc_models = {}
        self.load_rvc_models()
        
        # ゆっくりボイス識別プロファイル（F0範囲のヒューリスティック）
        self.speaker_profiles = {
            'reimu': {'f0_range': [180, 250], 'name': 'ゆっくり霊夢'},
            'marisa': {'f0_range': [150, 220], 'name': 'ゆっくり魔理沙'},
            'zundamon': {'f0_range': [200, 280], 'name': 'ずんだもん'},
        }

        # 合成音声→人っぽい声のマッピング管理
        self.mapping_manager = VoiceMappingManager()
    
    def load_rvc_models(self):
        """RVCモデルをロード"""
        models_dir = Path(__file__).parent / 'models' / 'rvc'
        repo_root = Path(__file__).parent.parent

        # scan both canonical models folder and repo root for loose files
        candidates = []
        if models_dir.exists():
            candidates += list(models_dir.glob('*.pth'))

        # also check repo root for any pth/index placed at project root
        candidates += list(repo_root.glob('*.pth'))

        if not candidates:
            logger.warning(f"No RVC .pth models found in {models_dir} or repo root {repo_root}")
            return

        logger.info(f"Scanning RVC models in {models_dir} and {repo_root}")
        for model_file in candidates:
            character = model_file.stem
            try:
                index_file = model_file.with_suffix('.index')
                self.rvc_models[character] = {
                    'model_path': str(model_file),
                    'index_path': str(index_file) if index_file.exists() else None,
                }
                logger.info(f"✓ Registered RVC model: {character} -> {model_file}")
            except Exception as e:
                logger.error(f"Failed to register {model_file}: {e}")
    
    def separate_vocals(self, audio_tensor):
        """Demucsで音声とBGMを分離
        
        Returns:
            (vocals, accompaniment): 音声とその他の音を分離したテンソル
        """
        if self.separator is None or not HAS_TORCH:
            logger.debug("Demucs not available, returning original audio")
            if HAS_TORCH:
                return audio_tensor, torch.zeros_like(audio_tensor)
            else:
                return audio_tensor, np.zeros_like(audio_tensor)
        
        try:
            with torch.no_grad():
                # Demucs は複数のステム（vocals, drums, bass, other）を返す
                sources = self.separator(audio_tensor.unsqueeze(0).to(self.device))
                # sources shape: (1, stems, samples)
                vocals = sources[0, 0].cpu()  # vocals stem
                # その他のステムを合成
                accompaniment = sources[0, 1:].sum(dim=0).cpu()
            return vocals, accompaniment
        except Exception as e:
            logger.error(f"Separation failed: {e}")
            if HAS_TORCH:
                return audio_tensor, torch.zeros_like(audio_tensor)
            else:
                return audio_tensor, np.zeros_like(audio_tensor)
    
    def detect_speakers(self, vocal_audio, threshold: float = 0.2) -> List[Tuple[str, float]]:
        """複数の候補話者を識別（フレームのF0ヒット率で判定）

        Args:
            vocal_audio: numpy array or torch.Tensor [samples]
            threshold: プロファイル範囲に入ったフレーム率のしきい値

        Returns:
            List[Tuple[str, float]]: 検出された候補話者ID（スコア順）
        """
        if not HAS_LIBROSA:
            logger.debug("librosa not available, returning default")
            return [('unknown', 0.5)]
        
        try:
            # numpy に変換
            if HAS_TORCH and torch.is_tensor(vocal_audio):
                audio_np = vocal_audio.numpy()
            else:
                audio_np = vocal_audio
            
            f0, voiced_flag, _ = librosa.pyin(
                audio_np,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            if f0 is None or voiced_flag is None:
                return []
            hits: List[Tuple[str, float]] = []
            total_frames = max(int(voiced_flag.sum()), 1)
            for sid, prof in self.speaker_profiles.items():
                fmin, fmax = prof['f0_range']
                in_range = ((f0 >= fmin) & (f0 <= fmax) & voiced_flag)
                score = float(np.count_nonzero(in_range)) / float(total_frames)
                hits.append((sid, float(score)))
            hits.sort(key=lambda x: x[1], reverse=True)
            return hits
        except Exception as e:
            logger.error(f"Multi speaker detection failed: {e}")
            return [('unknown', 0.1)]

    def detect_speakers(self, vocal_audio, threshold: float = 0.2) -> List[str]:
        """複数の候補話者を識別（フレームのF0ヒット率で判定）

        Args:
            vocal_audio: torch.Tensor [samples]
            threshold: プロファイル範囲に入ったフレーム率のしきい値

        Returns:
            List[str]: 検出された候補話者ID（スコア順）
        """
        try:
            audio_np = vocal_audio.numpy()
            f0, voiced_flag, _ = librosa.pyin(
                audio_np,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )
            if f0 is None or voiced_flag is None:
                return []
            hits: List[Tuple[str, float]] = []
            total_frames = max(int(voiced_flag.sum()), 1)
            for sid, prof in self.speaker_profiles.items():
                fmin, fmax = prof['f0_range']
                in_range = ((f0 >= fmin) & (f0 <= fmax) & voiced_flag)
                score = float(np.count_nonzero(in_range)) / float(total_frames)
                # スコアを記録（しきい値以下でも候補として返すことがある）
                hits.append((sid, float(score)))
            # スコア順にソートし、閾値未満のものも含めて返す（呼び出し側で閾値処理）
            hits.sort(key=lambda x: x[1], reverse=True)
            return hits
        except Exception as e:
            logger.error(f"Multi speaker detection failed: {e}")
            return []

    def diarize_segments(self, vocal_audio) -> List[Tuple[int, int]]:
        """簡易ダイアライゼーション（スタブ）。
        本格的な話者分離が必要なら pyannote.audio 等の導入を推奨。
        現在は全区間を1セグメントとして返す。
        Returns: List of (start_sample, end_sample)
        """
        return [(0, int(vocal_audio.shape[-1]))]
    
    def convert_voice(self, audio_np, speaker_id, target_speaker=None):
        """DSP ベースの簡易変換（ピッチシフト + ローパス）
        
        Args:
            audio_np: 音声データ（numpy array）
            speaker_id: 元の話者（識別結果）
            target_speaker: ターゲット話者（指定なしなら自然な声へ）
        
        Returns:
            converted: 変換後の音声（numpy array）
        """
        try:
            sr = self.sample_rate
            y = audio_np.astype(np.float32)

            # ピッチマッピング
            pitch_map = {
                'natural_female': -2.0,
                'natural_male': -4.0,
                'natural_boy': 3.0,
            }
            n_steps = pitch_map.get(target_speaker, -2.0) if isinstance(target_speaker, str) else -2.0

            # librosa が使える場合はピッチシフト
            if HAS_LIBROSA:
                try:
                    y = librosa.effects.pitch_shift(y, sr, n_steps=n_steps)
                except Exception as e:
                    logger.debug(f"Pitch shift failed: {e}")
            else:
                # フォールバック：簡易ピッチシフト（補間ベース）
                factor = 2 ** (n_steps / 12.0)
                n_new = int(len(y) / factor)
                if n_new > 0:
                    y = np.interp(np.linspace(0, len(y) - 1, n_new), np.arange(len(y)), y)

            # ローパスフィルタ
            try:
                nyq = 0.5 * sr
                cutoff = min(8000.0, nyq - 100.0)
                b, a = signal.butter(4, cutoff / nyq, btype='low')
                y = signal.filtfilt(b, a, y)
            except Exception as e:
                logger.debug(f"Lowpass failed: {e}")

            # 正規化
            maxv = np.max(np.abs(y)) if y.size else 0.0
            if maxv > 0:
                y = y / maxv * 0.98
            
            return y.astype(np.float32)
        except Exception as e:
            logger.error(f"convert_voice failed: {e}")
            return audio_np
    
    def process_chunk_streaming(self, audio_bytes, target_speaker=None):
        """ストリーミング用チャンク処理 - 軽量・高速
        
        受け取ったバイナリを自動判定：
        - WAV ファイル形式
        - raw float32
        - raw int16
        
        ストリーミングはリアルタイム性が最優先のため、
        ボーカル分離や複雑な話者識別は行わない。
        代わりに簡易的なピッチシフト処理のみ実施。
        
        Args:
            audio_bytes: バイナリ音声データ
            target_speaker: ターゲット話者指定
        
        Returns:
            numpy array (float32): 変換後のオーディオフレーム（-1 ～ 1）
        """
        try:
            logger.info(f"/convert: Received {len(audio_bytes)} bytes")
            
            # チャンク検証
            if len(audio_bytes) < 100:
                logger.warning(f"/convert: REJECTED - Chunk too small: {len(audio_bytes)} bytes (<100)")
                return np.zeros(50, dtype=np.float32)
            
            if len(audio_bytes) % 4 != 0 and len(audio_bytes) % 2 != 0:
                logger.warning(f"/convert: REJECTED - Odd byte count: {len(audio_bytes)} (not divisible by 2 or 4)")
                return np.zeros(50, dtype=np.float32)
            
            logger.info(f"/convert: Chunk validation passed, processing...")
            
            audio_np = None
            
            # 1. WAV ファイルとして試みる
            if len(audio_bytes) > 12:  # WAV ヘッダー最小サイズ
                try:
                    bio = io.BytesIO(audio_bytes)
                    data, sr = sf.read(bio, dtype='float32')
                    
                    # ステレオ → モノに変換
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    
                    # リサンプリング（必要に応じて）
                    if sr != self.sample_rate:
                        n_new = int(len(data) * self.sample_rate / sr)
                        if n_new > 0:
                            data = np.interp(
                                np.linspace(0, len(data) - 1, n_new),
                                np.arange(len(data)),
                                data
                            )
                    
                    audio_np = data.astype(np.float32)
                    logger.debug(f"Parsed as WAV: shape={audio_np.shape}")
                except Exception as e:
                    logger.debug(f"WAV parse failed: {type(e).__name__}")
            
            # 2. raw float32 として試みる
            if audio_np is None and len(audio_bytes) % 4 == 0:
                try:
                    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                    logger.debug(f"Parsed as float32: shape={audio_np.shape}")
                except Exception as e:
                    logger.debug(f"Float32 parse failed: {e}")
            
            # 3. raw int16 として試みる
            if audio_np is None and len(audio_bytes) % 2 == 0:
                try:
                    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_np = audio_int16.astype(np.float32) / 32768.0
                    logger.debug(f"Parsed as int16: shape={audio_np.shape}")
                except Exception as e:
                    logger.debug(f"Int16 parse failed: {e}")
            
            # 4. 最終フォールバック：zero padding して float32 に
            if audio_np is None:
                logger.warning(f"Could not parse {len(audio_bytes)} bytes, zero-padding to nearest float32 boundary")
                padded_len = ((len(audio_bytes) + 3) // 4) * 4
                padded = bytearray(padded_len)
                padded[:len(audio_bytes)] = audio_bytes
                try:
                    audio_np = np.frombuffer(bytes(padded), dtype=np.float32)
                except Exception as e:
                    logger.error(f"Even zero-padding failed: {e}")
                    # 最後の手段：全て 0
                    audio_np = np.zeros(4096, dtype=np.float32)

            if audio_np is None or audio_np.size == 0:
                audio_np = np.zeros(4096, dtype=np.float32)

            # ストリーミング用処理: 簡易ピッチシフト
            pitch_map = {
                'natural_female': -2.0,
                'natural_male': -4.0,
                'natural_boy': 3.0,
                'yukkuri': 0.0,
                'natural': -2.0,
            }
            n_steps = pitch_map.get(target_speaker, -2.0) if isinstance(target_speaker, str) else -2.0
            
            output = audio_np.copy()
            
            # librosa でピッチシフト
            if HAS_LIBROSA and n_steps != 0.0:
                try:
                    output = librosa.effects.pitch_shift(output, sr=self.sample_rate, n_steps=n_steps)
                except Exception as e:
                    logger.debug(f"Pitch shift failed: {e}")
                    output = audio_np

            # 簡易正規化
            max_val = np.abs(output).max()
            if max_val > 1.0:
                output = output / max_val * 0.95
            
            logger.info(f"/convert: SUCCESS - Processed {len(audio_np)} input samples -> {len(output)} output samples")
            return output.astype(np.float32)
        
        except Exception as e:
            logger.error(f"/convert: FAILED - {type(e).__name__}: {e}", exc_info=True)
            # フォールバック：無音を返す
            return np.zeros(50, dtype=np.float32)

    def process_chunk(self, audio_bytes, target_speaker=None):
        """音声チャンク（1秒分）をリアルタイム処理
        
        Args:
            audio_bytes: バイナリ音声データ
            target_speaker: ターゲット話者指定
        
        Returns:
            (converted_bytes, metadata): 変換後の音声とメタデータ
        """
        try:
            # バイトデータを WAV として解釈できるか試みる（POST で WAV ファイル送信に対応）
            try:
                bio = io.BytesIO(audio_bytes)
                data, sr = sf.read(bio, dtype='float32')
                if sr != self.sample_rate:
                    data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                audio_np = data.astype(np.float32)
            except Exception:
                # 生の float32 バッファを想定
                audio_np = np.frombuffer(audio_bytes, dtype=np.float32)

            audio_tensor = torch.from_numpy(audio_np)
            # 入力の RMS をログ
            try:
                in_rms = float(np.sqrt(np.mean(np.square(audio_np))))
                logger.info(f"Input RMS: {in_rms:.6f} (samples={audio_np.size})")
            except Exception:
                in_rms = None
            
            # 1. 音声とBGMを分離
            vocals, accompaniment = self.separate_vocals(audio_tensor)
            
            # 2. ゆっくり話者を識別（複数候補: (id, score) リスト）
            raw_candidates = self.detect_speakers(vocals, threshold=0.0)
            # raw_candidates: List[Tuple[id, score]]

            converted_vocals = None
            final_target: Optional[str] = None

            if raw_candidates:
                # スコアを正規化して重みを算出
                ids, scores = zip(*raw_candidates)
                scores = np.array(scores, dtype=float)
                # しきい値未満は無視（0.05 以下はノイズ扱い）
                scores[scores < 0.05] = 0.0
                total = float(scores.sum())
                if total <= 0:
                    # 候補があるがスコアが小さい -> unknown 扱い
                    candidates = []
                else:
                    weights = scores / total
                    candidates = list(zip(ids, weights))

                # 各候補ごとにマッピングを確認して変換（無ければ原音）
                if candidates:
                    vocals_np = vocals.numpy()
                    mixed = np.zeros_like(vocals_np)
                    for sid, weight in candidates:
                        mapped_target = self.mapping_manager.get_target_for(sid)
                        use_target = target_speaker or mapped_target
                        if use_target:
                            converted_np = self.convert_voice(vocals_np, sid, use_target)
                        else:
                            converted_np = vocals_np
                        mixed += converted_np * float(weight)
                    converted_vocals = torch.from_numpy(mixed)
                    # 最上位候補のターゲットを報告
                    final_target = self.mapping_manager.get_target_for(ids[0])
            
            if converted_vocals is None:
                # 変換適用対象が無ければ元の音声を使う
                converted_vocals = vocals
            
            # 4. BGMと再合成
            output = converted_vocals + accompaniment
            
            # 5. 正規化
            max_val = torch.abs(output).max()
            if max_val > 1.0:
                output = output / max_val
            
            # メタデータ
            # メタデータ: candidates は (id, weight) のリスト
            metadata = {
                'speaker': ids[0] if raw_candidates else 'unknown',
                'speaker_name': self.speaker_profiles.get(
                    (ids[0] if raw_candidates else 'unknown'), {}
                ).get('name', 'Unknown'),
                'timestamp': datetime.now().isoformat(),
                'target': final_target if final_target else None,
                'candidates': [{'id': sid, 'weight': float(w)} for sid, w in (candidates if 'candidates' in locals() and candidates else [])],
            }
            
            # 出力を WAV バイナリに変換して返す
            out_np = output.cpu().numpy().astype(np.float32)
            # ensure mono
            if out_np.ndim > 1:
                out_np = np.mean(out_np, axis=0)
            try:
                out_rms = float(np.sqrt(np.mean(np.square(out_np))))
                logger.info(f"Output RMS: {out_rms:.6f}")
            except Exception:
                out_rms = None
            # デバッグ: 最後の出力を tmp/last_out.wav に保存（トラブルシュート用）
            try:
                tmp_dir = Path(__file__).parent / 'tmp'
                tmp_dir.mkdir(parents=True, exist_ok=True)
                sf.write(str(tmp_dir / 'last_out.wav'), out_np, self.sample_rate, subtype='PCM_16')
            except Exception as e:
                logger.warning(f"Failed to write last_out.wav: {e}")
            buf = io.BytesIO()
            sf.write(buf, out_np, self.sample_rate, subtype='PCM_16')
            buf.seek(0)
            return buf.read(), metadata
        
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise


# グローバルコンバーター
converter = None


def init_converter():
    """コンバーターを初期化"""
    global converter
    if converter is None:
        converter = VoiceConverter()
        logger.info("✓ Voice converter initialized")


# (VoiceMappingManager はファイル先頭に1つだけ定義されています)


# ========== HTTP エンドポイント ==========

@app.route('/health', methods=['GET'])
def health_check():
    """ヘルスチェック"""
    return jsonify({
        'status': 'ok',
        'device': CONFIG['DEVICE'],
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'models_available': list(converter.rvc_models.keys()),
        'speakers': list(converter.speaker_profiles.keys()),
    })


@app.route('/convert', methods=['POST'])
def convert():
    """
    音声変換エンドポイント（ストリーミング）
    
    Request:
        - audio_data (binary): Float32 PCM, 16kHz mono or WAV file
        - target_speaker (query param): ターゲット話者（オプション）
    
    Response:
        - audio_data (binary): 変換後の PCM フレーム（int16、ストリーミング）
    """
    try:
        audio_data = request.data
        clen = len(audio_data)
        logger.info(f"/convert called. bytes={clen}")
        
        if not audio_data:
            logger.warning("/convert: no audio data")
            return jsonify({'error': 'No audio data'}), 400
        
        target_speaker = request.args.get('target', 'natural')
        
        # 処理（チャンクごと）
        converted_np = converter.process_chunk_streaming(audio_data, target_speaker)
        
        # PCM int16 バイナリとして返す
        if converted_np is None or converted_np.size == 0:
            return jsonify({'error': 'No output'}), 500
        
        # -1.0 ～ 1.0 float32 から -32768 ～ 32767 int16 へ変換
        pcm_int16 = np.clip(converted_np * 32767.0, -32768, 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()
        
        response = send_file(
            io.BytesIO(pcm_bytes),
            mimetype='audio/wav'
        )
        response.headers['Content-Type'] = 'audio/wav'
        
        logger.info(f"/convert response size={len(pcm_bytes)} bytes")
        return response
    
    except Exception as e:
        logger.error(f"Error in /convert: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/process_chunk', methods=['POST'])
def process_chunk():
    """
    /convert のエイリアス（Chrome 拡張互換性）
    """
    return convert()


@app.route('/speakers', methods=['GET'])
def get_speakers():
    """利用可能な話者一覧を取得"""
    return jsonify({
        'speakers': [
            {
                'id': sid,
                'name': profile.get('name', sid),
                'f0_range': profile['f0_range'],
            }
            for sid, profile in converter.speaker_profiles.items()
        ]
    })


@app.route('/info', methods=['GET'])
def get_info():
    """システム情報を取得"""
    return jsonify({
        'project': 'Yukkuri Voice Converter',
        'version': '1.0.0',
        'description': 'Real-time Yukkuri voice to natural voice conversion',
        'backend': 'Demucs + RVC',
        'device': CONFIG['DEVICE'],
        'sample_rate': CONFIG['SAMPLE_RATE'],
    })


@app.route('/mapping', methods=['GET'])
def get_mapping():
    """合成音声→ターゲット音声の現在のマッピングを返す"""
    return jsonify({
        'mapping': converter.mapping_manager.get_mapping(),
        'available_sources': list(converter.speaker_profiles.keys()),
        'available_targets_hint': list(converter.rvc_models.keys()) or [
            'natural_female', 'natural_male', 'natural_boy'
        ],
    })


@app.route('/mapping', methods=['POST'])
def update_mapping():
    """マッピングを更新する
    Request JSON 例: {"reimu": "natural_female"}
    """
    try:
        data = request.get_json(silent=True) or {}
        if not isinstance(data, dict) or not data:
            return jsonify({'error': 'Invalid JSON'}), 400
        # validate: target が空文字は削除扱い
        # validate keys against known speakers
        allowed = list(converter.speaker_profiles.keys())
        cleaned = converter.mapping_manager.validate_update(data, allowed)
        if not cleaned:
            return jsonify({'error': 'no valid mapping entries provided'}), 400
        converter.mapping_manager.update(cleaned)
        return jsonify({'status': 'ok', 'mapping': converter.mapping_manager.get_mapping()})
    except Exception as e:
        logger.error(f"/mapping update failed: {e}")
        return jsonify({'error': str(e)}), 500


# ---- 簡易トレーニングAPI（スタブ） ----
TRAIN_JOBS = {}


@app.route('/train/upload', methods=['POST'])
def train_upload():
    """音声サンプルをアップロードして保存（スタブ）。multipart/form-data に対応"""
    try:
        up_dir = Path(__file__).parent / 'data' / 'uploads' / datetime.now().strftime('%Y%m%d_%H%M%S')
        up_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for key in request.files:
            f = request.files[key]
            dest = up_dir / f.filename
            f.save(dest)
            saved.append(str(dest))
        return jsonify({'status': 'ok', 'saved': saved})
    except Exception as e:
        logger.error(f"/train/upload failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/train/start', methods=['POST'])
def train_start():
    """トレーニング開始（スタブ）。実装は別スクリプト想定。"""
    try:
        payload = request.get_json(silent=True) or {}
        synthetic_id = payload.get('synthetic_id')
        if not synthetic_id:
            return jsonify({'error': 'synthetic_id required'}), 400
        job_id = datetime.now().strftime('%Y%m%d%H%M%S')
        TRAIN_JOBS[job_id] = {'status': 'queued', 'synthetic_id': synthetic_id, 'progress': 0}

        # Start background training thread
        def _bg_train(jid, sid):
            try:
                from backend.train.train_voice_model import run_training
                run_training(jid, sid, str(Path(__file__).parent / 'data' / 'uploads'), TRAIN_JOBS)
            except Exception as e:
                TRAIN_JOBS[jid]['status'] = 'failed'
                TRAIN_JOBS[jid]['error'] = str(e)

        t = Thread(target=_bg_train, args=(job_id, synthetic_id), daemon=True)
        t.start()

        return jsonify({'status': 'queued', 'job_id': job_id})
    except Exception as e:
        logger.error(f"/train/start failed: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/train/status', methods=['GET'])
def train_status():
    job_id = request.args.get('job_id')
    if not job_id:
        return jsonify({'error': 'job_id required'}), 400
    info = TRAIN_JOBS.get(job_id)
    if not info:
        return jsonify({'error': 'job not found'}), 404
    return jsonify({'status': info['status'], 'job': info})


@app.route('/mapping', methods=['DELETE'])
def delete_mapping():
    """key を指定してマッピングを削除する: /mapping?key=reimu"""
    key = request.args.get('key')
    if not key:
        return jsonify({'error': 'key required'}), 400
    mapping = converter.mapping_manager.get_mapping()
    if key not in mapping:
        return jsonify({'error': 'key not found'}), 404
    mapping.pop(key, None)
    converter.mapping_manager.update(mapping)
    return jsonify({'status': 'ok', 'mapping': converter.mapping_manager.get_mapping()})


@app.route('/models/refresh', methods=['POST'])
def refresh_models():
    """Reload RVC model scan and return available models."""
    try:
        converter.load_rvc_models()
        return jsonify({'status': 'ok', 'models': list(converter.rvc_models.keys())})
    except Exception as e:
        logger.error(f"/models/refresh failed: {e}")
        return jsonify({'error': str(e)}), 500


# ========== メイン ==========

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None, help='force device (cpu/cuda)')
    args = parser.parse_args()

    # ディレクトリ作成
    Path('./models/rvc').mkdir(parents=True, exist_ok=True)

    # device override
    if args.device:
        if args.device == 'cuda' and not torch.cuda.is_available():
            logger.warning('CUDA requested but not available; falling back to CPU')
            CONFIG['DEVICE'] = 'cpu'
        else:
            CONFIG['DEVICE'] = args.device

    logger.info(f"Using device: {CONFIG['DEVICE']}")

    # コンバーター初期化
    init_converter()

    # サーバー起動
    logger.info("=" * 60)
    logger.info("Starting Yukkuri Voice Converter Backend")
    logger.info("=" * 60)
    logger.info(f"Server: http://{args.host}:{args.port}")
    logger.info(f"Health check: http://localhost:{args.port}/health")
    logger.info(f"Speakers: http://localhost:{args.port}/speakers")
    logger.info("=" * 60)

    app.run(host=args.host, port=args.port, debug=False, threaded=True)

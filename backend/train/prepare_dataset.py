"""
prepare_dataset.py
前処理: アップロードされた音声を 16kHz モノラルにリサンプル・正規化し processed ディレクトリに保存。
出力: processed WAV と metadata.json
"""
import argparse
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import json


def prepare(input_dir: Path, output_dir: Path, sr: int = 16000):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = output_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    for p in sorted(input_dir.glob('*')):
        if not p.is_file():
            continue
        try:
            y, _ = librosa.load(str(p), sr=sr, mono=True)
            # トリミング
            y, _ = librosa.effects.trim(y, top_db=20)
            # 正規化
            if np.max(np.abs(y)) > 0:
                y = y / np.max(np.abs(y))
            out_path = processed_dir / (p.stem + '_processed.wav')
            sf.write(str(out_path), y, sr, subtype='FLOAT')
            metadata.append({
                'input': str(p),
                'output': str(out_path),
                'duration': float(len(y) / sr),
                'sr': sr
            })
        except Exception as e:
            print(f"Failed to process {p}: {e}")

    meta_path = output_dir / 'metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Prepared {len(metadata)} files -> {processed_dir}")
    return processed_dir, meta_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sr', type=int, default=16000)
    args = parser.parse_args()
    prepare(Path(args.input), Path(args.output), args.sr)


if __name__ == '__main__':
    main()

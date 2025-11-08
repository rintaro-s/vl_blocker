#!/usr/bin/env python3
"""auto_prepare_and_train.py

簡単な用途:
- mp3 / mp4 / wav 等のファイルを受け取り、ffmpeg を使って 16kHz, mono の WAV に変換
- 無音トリム・ピーク正規化・分割（固定長クリップ）を行い `output/<synthetic_id>/processed/` に書き出す
- metadata.json を出力する
- オプションでトレーニングスタブ (`train_voice_model.py`) を呼び出して学習ワークフローを起動する

依存: ffmpeg が PATH にあること。Python ライブラリは既存のプロジェクト要件（librosa, soundfile, numpy 等）を使います。

使い方 (PowerShell 例):
python backend\train\auto_prepare_and_train.py --inputs "C:\path\to\file1.mp3" "C:\path\to\dir_with_files\" --synthetic_id reimu --output_dir ..\data\prepared --clip_len 5 --run_train

"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import librosa
import numpy as np
import soundfile as sf


def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False


def collect_inputs(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for p in inputs:
        pth = Path(p)
        if pth.is_dir():
            for ext in ("*.mp3", "*.mp4", "*.wav", "*.m4a", "*.flac", "*.ogg"):
                for f in sorted(pth.glob(ext)):
                    paths.append(f)
        elif pth.exists():
            paths.append(pth)
        else:
            # allow glob patterns
            for f in sorted(Path('.').glob(p)):
                paths.append(f)
    return paths


def ffmpeg_to_wav(input_path: Path, out_wav: Path, sr: int = 16000) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        str(sr),
        "-ac",
        "1",
        "-vn",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)


def normalize_peak(y: np.ndarray, peak: float = 0.98) -> np.ndarray:
    maxv = float(np.max(np.abs(y))) if y.size else 0.0
    if maxv <= 0:
        return y
    return y / maxv * peak


def split_and_save(y: np.ndarray, sr: int, clip_len: float, min_clip: float, outdir: Path, base_name: str, metadata: list):
    clip_samples = int(round(clip_len * sr))
    n = len(y)
    idx = 0
    start = 0
    while start < n:
        end = start + clip_samples
        chunk = y[start:end]
        dur = len(chunk) / sr
        if dur >= min_clip:
            fname = f"{base_name}_{idx:04d}.wav"
            outpath = outdir / fname
            sf.write(str(outpath), chunk, sr, subtype='PCM_16')
            metadata.append({
                "file": str(outpath.relative_to(outdir.parent)),
                "duration": dur,
                "source_base": base_name,
            })
            idx += 1
        start = end


def process_file(p: Path, out_base: Path, synthetic_id: str, clip_len: float, min_clip: float, sr: int, trim_db: int, normalize: bool, metadata: list):
    # prepare temp wav via ffmpeg
    with tempfile.TemporaryDirectory() as td:
        tmpwav = Path(td) / (p.stem + ".wav")
        ffmpeg_to_wav(p, tmpwav, sr=sr)
        # load
        y, _ = librosa.load(str(tmpwav), sr=sr, mono=True)
        # trim silence
        if trim_db is not None:
            y, _ = librosa.effects.trim(y, top_db=trim_db)
        # normalize
        if normalize:
            y = normalize_peak(y)
        processed_dir = out_base / synthetic_id / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        base_name = p.stem
        split_and_save(y, sr, clip_len, min_clip, processed_dir, base_name, metadata)


def run_training_stub(prepared_dir: Path, synthetic_id: str) -> int:
    # Attempts to call train_voice_model.py with arguments. Uses same Python interpreter.
    script = Path(__file__).parent / "train_voice_model.py"
    if not script.exists():
        print("train_voice_model.py not found; training not started.")
        return 1
    cmd = [sys.executable, str(script), "--input_dir", str(prepared_dir), "--synthetic_id", synthetic_id]
    print("Running training stub:", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", "-i", nargs="+", required=True, help="Input files or directories (supports globs)")
    parser.add_argument("--synthetic_id", required=True)
    parser.add_argument("--output_dir", "-o", default="..\\data\\prepared", help="Base output directory")
    parser.add_argument("--clip_len", type=float, default=5.0, help="Clip length in seconds (default 5)")
    parser.add_argument("--min_clip", type=float, default=2.0, help="Minimum clip length to keep (default 2)")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--trim_db", type=int, default=30, help="top_db for librosa.effects.trim (default 30)")
    parser.add_argument("--no_normalize", action="store_true", help="Disable peak normalization")
    parser.add_argument("--run_train", action="store_true", help="After prepare, call training stub")
    args = parser.parse_args(argv)

    if not check_ffmpeg():
        print("ffmpeg not found in PATH. Please install ffmpeg and ensure it's on PATH.")
        sys.exit(2)

    input_paths = collect_inputs(args.inputs)
    if not input_paths:
        print("No input files found.")
        sys.exit(3)

    out_base = Path(args.output_dir)
    metadata = []
    for p in input_paths:
        try:
            print(f"processing: {p}")
            process_file(p, out_base, args.synthetic_id, args.clip_len, args.min_clip, args.sr, args.trim_db if args.trim_db >= 0 else None, not args.no_normalize, metadata)
        except Exception as e:
            print(f"Failed processing {p}: {e}")

    # write metadata
    meta_path = out_base / args.synthetic_id / "metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("Prepared clips:", len(metadata))
    print("Metadata written to:", meta_path)

    if args.run_train:
        prepared_dir = out_base / args.synthetic_id / "processed"
        rc = run_training_stub(prepared_dir, args.synthetic_id)
        if rc == 0:
            print("Training stub finished successfully.")
        else:
            print("Training stub returned code", rc)


if __name__ == "__main__":
    main()

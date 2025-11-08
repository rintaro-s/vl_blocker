"""
train_stub.py

Lightweight helper that shows how to prepare training data / model folders from
user-provided samples. This is a *stub* and intentionally does not run heavy ML
training in-repo. Instead it prepares directories and example metadata so users
can later run RVC training or other frameworks.

Example:
  python train_stub.py --character reimu --samples ./samples/reimu/*.wav

The script will:
 - create backend/models/synthetic/reimu/
 - copy sample WAVs (user must ensure they are correct sample rate / mono)
 - write a meta.json describing the dataset

For full training, see README instructions about using external RVC training tools.
"""
import argparse
from pathlib import Path
import shutil
import json


def prepare(character: str, samples: list, base_dir: Path):
    dest = base_dir / 'models' / 'synthetic' / character
    dest.mkdir(parents=True, exist_ok=True)
    sample_dir = dest / 'samples'
    sample_dir.mkdir(exist_ok=True)
    for s in samples:
        s_path = Path(s)
        if s_path.exists():
            shutil.copy2(s_path, sample_dir / s_path.name)
    meta = {
        'id': character,
        'num_samples': len(list(sample_dir.iterdir())),
        'notes': 'Prepared by train_stub; run full training externally.'
    }
    (dest / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Prepared {character} -> {dest} (samples: {meta["num_samples"]})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--character', required=True)
    p.add_argument('--samples', nargs='+', required=True)
    p.add_argument('--base', default=Path(__file__).parent)
    args = p.parse_args()
    prepare(args.character, args.samples, Path(args.base))

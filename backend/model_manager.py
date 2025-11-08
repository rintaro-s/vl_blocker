"""
Simple model manager for the project.
Provides utilities to register, list and load model metadata.
This is a lightweight helper (no heavy ML). Models are represented by files
under backend/models/{type}/ and a metadata JSON file.

Usage:
  from model_manager import ModelManager
  mm = ModelManager('e:/github/yukkuri_blocker/backend/models')
  mm.scan()
  mm.list_models()

Note: actual model loading (RVC, classifier) should be implemented in
specific loader functions depending on the model format.
"""
import json
from pathlib import Path
from typing import Dict, List


class ModelManager:
    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        # expected layout: base/synthetic/  base/target/
        self.kinds = ['synthetic', 'target']
        for k in self.kinds:
            (self.base / k).mkdir(exist_ok=True)
        self._models: Dict[str, List[Dict]] = {k: [] for k in self.kinds}

    def scan(self):
        """Scan model directories and load metadata if present."""
        for k in self.kinds:
            lst = []
            for p in sorted((self.base / k).iterdir()):
                if p.is_dir():
                    meta = {}
                    metaf = p / 'meta.json'
                    if metaf.exists():
                        try:
                            meta = json.loads(metaf.read_text(encoding='utf-8'))
                        except Exception:
                            meta = {}
                    else:
                        # minimal metadata
                        meta = {'id': p.name, 'path': str(p)}
                    lst.append(meta)
            self._models[k] = lst

    def list_models(self, kind: str = None) -> Dict:
        if kind:
            return {kind: self._models.get(kind, [])}
        return self._models

    def register_model(self, kind: str, model_id: str, model_dir: str, metadata: Dict = None):
        """Register a model directory by writing meta.json. Expects model_dir already created.
        kind: 'synthetic' or 'target'
        model_dir: path to the model folder (will be copied/moved by user)
        metadata: dict to write into meta.json
        """
        if kind not in self.kinds:
            raise ValueError('unknown kind')
        target = self.base / kind / model_id
        target.mkdir(parents=True, exist_ok=True)
        # we don't move files here; user should place files in this folder.
        meta = metadata or {}
        meta.setdefault('id', model_id)
        meta.setdefault('path', str(target))
        (target / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        self.scan()
        return meta


if __name__ == '__main__':
    mm = ModelManager(Path(__file__).parent / 'models')
    mm.scan()
    print('Models:')
    print(mm.list_models())

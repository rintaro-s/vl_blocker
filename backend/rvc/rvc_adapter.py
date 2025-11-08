import torch
import numpy as np
from pathlib import Path
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class RVCManager:
    """RVC adaptor that expects TorchScript models placed under `models/rvc/<id>.<ext>`.

    Supported formats (best-effort):
    - TorchScript saved module: `models/rvc/<id>.jit` or `<id>.pt` or `<id>.pth` (if TorchScript).

    If a model cannot be loaded, the manager leaves it out and convert will fallback to pass-through.
    """

    def __init__(self, models_dir: Optional[str] = None, device: Optional[torch.device] = None):
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent.parent / 'models' / 'rvc'
        self.device = device or torch.device('cpu')
        self.models = {}  # id -> scripted module
        self.scan_models()

    def scan_models(self):
        if not self.models_dir.exists():
            logger.info(f"RVC models dir not found: {self.models_dir}")
            return
        logger.info(f"Scanning RVC models in {self.models_dir}")
        for f in sorted(self.models_dir.iterdir()):
            if not f.is_file():
                continue
            name = f.stem
            if f.suffix.lower() in ('.jit', '.pt', '.pth'):
                try:
                    # Try to load as TorchScript
                    mod = torch.jit.load(str(f), map_location=self.device)
                    mod.eval()
                    self.models[name] = mod
                    logger.info(f"Loaded RVC scripted model: {name} from {f}")
                except Exception as e:
                    logger.warning(f"Failed to torch.jit.load {f}: {e}")

    def list_models(self) -> List[str]:
        return list(self.models.keys())

    def has_model(self, model_id: str) -> bool:
        return model_id in self.models

    def infer(self, audio_np: np.ndarray, model_id: str, sr: int = 16000) -> np.ndarray:
        """Run inference using the scripted model.

        audio_np: 1-D float32 numpy array in [-1,1]
        returns: 1-D float32 numpy array
        """
        if model_id not in self.models:
            raise RuntimeError(f"Model {model_id} not loaded")
        mod = self.models[model_id]
        try:
            # Prepare tensor: (1, samples)
            x = torch.from_numpy(audio_np.astype(np.float32)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = mod(x)
            # Accept tensor or tuple results
            if isinstance(out, (tuple, list)):
                out = out[0]
            out_np = out.squeeze(0).cpu().numpy().astype(np.float32)
            return out_np
        except Exception as e:
            logger.error(f"RVC inference failed for {model_id}: {e}")
            raise

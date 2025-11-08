"""
RVC adapter helpers.

This module provides a small adapter layer that attempts to run inference
from either a runnable torch module or a saved state_dict. It is intentionally
lightweight: if no concrete architecture/inference implementation is present
in the repository, these helpers will act as graceful no-ops and log useful
diagnostic information for the user.

Implementers can extend this file to add specific RVC model constructors or
bridge code to third-party inference utilities.
"""
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


def infer_from_state_dict(state_dict: dict, audio_np: np.ndarray, device=None):
    """Best-effort: attempt to infer from a raw state_dict.

    Currently this function does not implement a generic RVC loader because
    architecture details vary across model artifacts. It returns None to
    indicate that no inference was performed.

    You can extend this function to instantiate a known architecture and
    load the state_dict, then run inference.
    """
    if not isinstance(state_dict, dict):
        return None

    logger.info("rvc_adapter: infer_from_state_dict called: keys=%d", len(state_dict.keys()))

    # Detect placeholders created by the training stub
    try:
        # state_dict might actually be raw bytes if the file was a placeholder
        if isinstance(state_dict, bytes) or any(isinstance(v, (bytes, bytearray)) for v in state_dict.values()):
            logger.info("rvc_adapter: state_dict contains raw bytes or placeholder; skipping inference")
            return None
    except Exception:
        pass

    # No generic loader implemented here
    logger.info("rvc_adapter: no generic RVC loader implemented; please provide a runnable model artifact or extend this adapter")
    return None


def inspect_module(mod):
    """Return a short diagnostic dict for a loaded module/object."""
    info = {
        'repr': repr(mod)[:200],
        'has_attrs': {},
        'dir': []
    }
    try:
        attrs = ['infer', 'forward', '__call__', 'to', 'eval']
        info['has_attrs'] = {a: hasattr(mod, a) for a in attrs}
        try:
            info['dir'] = [x for x in dir(mod) if not x.startswith('_')][:40]
        except Exception:
            info['dir'] = []
    except Exception:
        pass
    return info

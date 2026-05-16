"""
gpu_utils.py — minimal CuPy/NumPy dispatch.

Tries to import CuPy; if unavailable or no GPU is present, falls back to NumPy
silently. Use `get_xp(prefer_gpu=True)` to obtain the array module, and
`to_numpy(arr)` to bring a result back to host memory.
"""

from __future__ import annotations

import numpy as _np

_CUPY = None
_GPU_OK = False
_GPU_STATUS = "GPU disabled: CuPy not imported"

try:  # pragma: no cover - environment dependent
    import cupy as _cp  # type: ignore

    try:
        # Probe a device. If no CUDA device is available this will raise.
        _n_dev = _cp.cuda.runtime.getDeviceCount()
        if _n_dev > 0:
            _dev = _cp.cuda.Device(0)
            _name = _cp.cuda.runtime.getDeviceProperties(0).get("name", b"GPU")
            if isinstance(_name, bytes):
                _name = _name.decode("utf-8", errors="replace")
            _CUPY = _cp
            _GPU_OK = True
            _GPU_STATUS = f"GPU enabled: CuPy {_cp.__version__} / device 0 = {_name}"
        else:
            _GPU_STATUS = "GPU disabled: no CUDA devices found"
    except Exception as _exc:
        _GPU_STATUS = f"GPU disabled: CuPy import OK but no usable device ({_exc})"
except Exception as _exc:
    _GPU_STATUS = f"GPU disabled: CuPy not available ({_exc})"


def get_xp(prefer_gpu: bool = True):
    """Return the array module to use: cupy if available and prefer_gpu, else numpy."""
    if prefer_gpu and _GPU_OK and _CUPY is not None:
        return _CUPY
    return _np


def to_numpy(arr):
    """Move array to host (NumPy). Pass-through for NumPy arrays."""
    if arr is None:
        return None
    if _CUPY is not None and isinstance(arr, _CUPY.ndarray):  # type: ignore[attr-defined]
        return _CUPY.asnumpy(arr)
    return _np.asarray(arr)


def gpu_info() -> str:
    """Return a one-line description of GPU availability."""
    return _GPU_STATUS


def gpu_available() -> bool:
    return _GPU_OK

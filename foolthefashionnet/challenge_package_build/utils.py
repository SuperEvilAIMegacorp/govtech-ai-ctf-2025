# utils.py
from __future__ import annotations
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def linf(delta: np.ndarray, eps: float | None = None) -> tuple[float, bool]:
    """
    Compute L∞(delta). If eps is provided, also return whether it passes.
    delta: any float-like ndarray. Works with shapes (1,28,28), (28,28), (H,W,C), etc.
    """
    if not isinstance(delta, np.ndarray):
        delta = np.asarray(delta)
    if delta.dtype != np.float32 and delta.dtype != np.float64:
        delta = delta.astype(np.float32, copy=False)
    mx = float(np.max(np.abs(delta)))
    return mx, (mx <= eps + 1e-8) if eps is not None else True

def ssim_01(a: np.ndarray, b: np.ndarray) -> float:
    """
    SSIM for images scaled to [0,1].
    Accepts grayscale (H,W) or color (H,W,C). Returns scalar float.
    """
    if not isinstance(a, np.ndarray): a = np.asarray(a)
    if not isinstance(b, np.ndarray): b = np.asarray(b)
    if a.dtype != np.float32 and a.dtype != np.float64: a = a.astype(np.float32, copy=False)
    if b.dtype != np.float32 and b.dtype != np.float64: b = b.astype(np.float32, copy=False)

    # Clip to [0,1] defensively
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)

    # skimage>=0.19 uses channel_axis instead of multichannel
    if a.ndim == 3 and b.ndim == 3:
        return float(ssim(a, b, data_range=1.0, channel_axis=-1))
    else:
        return float(ssim(a, b, data_range=1.0))


"""Image Enhancer v2 — core logic.

Pipeline (gentle by design):
  1. Light denoise BEFORE upscale (conservative fastNlMeans)
  2. Real-ESRGAN realesr-general-x4v3, tiled CPU inference (400px tiles)
  3. Very light unsharp-mask sharpening (toggle, default ON, no halos)
  4. Auto-contrast OFF by default (toggle available)
Output longest side capped at 6000px (free-tier memory safety).
"""
import os
import cv2
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'realesr-general-x4v3.pth')

SCALE = 4
TILE = 400
OVERLAP = 10
HALO = 16
MAX_OUT = 6000  # longest output side

_model = None


def _get_model():
    global _model
    if _model is None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class SRVGGNetCompact(nn.Module):
            def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                         num_conv=32, upscale=4):
                super().__init__()
                self.upscale = upscale
                layers = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
                          nn.PReLU(num_parameters=num_feat)]
                for _ in range(num_conv):
                    layers += [nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                               nn.PReLU(num_parameters=num_feat)]
                layers += [nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1)]
                self.body = nn.Sequential(*layers)
                self.upsampler = nn.PixelShuffle(upscale)

            def forward(self, x):
                out = self.upsampler(self.body(x))
                base = F.interpolate(x, scale_factor=self.upscale, mode='nearest')
                return out + base

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f'Model file missing: {MODEL_PATH}')
        torch.set_num_threads(2)
        m = SRVGGNetCompact()
        sd = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        params = sd['params'] if 'params' in sd else sd
        # ckpt keys are body.N.* matching our Sequential layout exactly
        m.load_state_dict(params, strict=True)
        m.eval()
        _model = m
    return _model


def _upscale_tiled(rgb):
    """rgb: HxWx3 uint8. Returns H*4 x W*4 uint8 via tiled inference."""
    import torch
    model = _get_model()
    h, w = rgb.shape[:2]
    out = np.zeros((h * SCALE, w * SCALE, 3), np.uint8)
    stride = TILE - 2 * OVERLAP
    with torch.no_grad():
        for y0 in range(0, h, stride):
            for x0 in range(0, w, stride):
                y1, x1 = min(y0 + TILE, h), min(x0 + TILE, w)
                hy0, hx0 = max(y0 - HALO, 0), max(x0 - HALO, 0)
                hy1, hx1 = min(y1 + HALO, h), min(x1 + HALO, w)
                patch = rgb[hy0:hy1, hx0:hx1]
                cy0, cx0 = y0 - hy0, x0 - hx0
                t = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                sr = model(t).squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy()
                sr = (sr * 255.0 + 0.5).astype(np.uint8)
                th, tw = (y1 - y0), (x1 - x0)
                core = sr[cy0 * SCALE:(cy0 + th) * SCALE, cx0 * SCALE:(cx0 + tw) * SCALE]
                out[y0 * SCALE:y1 * SCALE, x0 * SCALE:x1 * SCALE] = core
    return out


def _light_unsharp(rgb, amount=0.35):
    blur = cv2.GaussianBlur(rgb, (0, 0), 2.0)
    sharp = cv2.addWeighted(rgb, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def enhance(pil_img, do_sharpen=True, do_contrast=False):
    """Full v2 pipeline. Returns (PIL result, note_str or None)."""
    rgb = np.array(pil_img.convert('RGB'))
    h, w = rgb.shape[:2]
    note = None

    # Safety cap: pre-shrink so output longest side <= MAX_OUT
    if max(h, w) * SCALE > MAX_OUT:
        s = MAX_OUT / (max(h, w) * SCALE)
        rgb = np.array(Image.fromarray(rgb).resize((int(w * s), int(h * s)), Image.LANCZOS))
        h, w = rgb.shape[:2]
        note = (f'Note: bari image ko pehle {w}x{h} kiya taake output {MAX_OUT}px '
                f'se zyada na ho. / Large image pre-shrunk for memory safety.')

    # 1. light denoise BEFORE upscale (conservative)
    rgb = cv2.fastNlMeansDenoisingColored(rgb, None, h=3, hColor=3,
                                          templateWindowSize=7, searchWindowSize=21)
    # 2. AI upscale x4
    rgb = _upscale_tiled(rgb)
    # 3. very light sharpening (default ON, subtle — no halos)
    if do_sharpen:
        rgb = _light_unsharp(rgb, amount=0.35)
    # 4. auto-contrast OFF by default
    if do_contrast:
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
        rgb = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    return Image.fromarray(rgb), note

import numpy as np
from PIL import Image, ImageFilter
def to_f(im): return np.asarray(im.convert("RGB"), np.float32) / 255.0
def to_im(a): return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))
def resize_f(a, size):
    # per-channel float resize via PIL 'F' mode keeps precision
    return np.stack([np.asarray(Image.fromarray(a[..., c]).resize(size, Image.LANCZOS)) for c in range(3)], -1)
def finish(src, ai, strength=1.0, color_lock=True, grain=0.0, seed=0):
    W, H = ai.size
    a = to_f(ai)
    base = resize_f(to_f(src), (W, H))
    out = base + strength * (a - base)
    if color_lock:
        sw, sh = max(8, W // 32), max(8, H // 32)
        corr = resize_f(base, (sw, sh)) - resize_f(out, (sw, sh))
        out = out + resize_f(corr, (W, H))
    if grain > 0:
        rng = np.random.default_rng(seed)
        n = rng.standard_normal((H, W)).astype(np.float32)
        k = np.array([0.2, 0.6, 0.2], np.float32)  # small blur so grain isn't single-pixel noise
        n = k[0]*np.roll(n, 1, 0) + k[1]*n + k[2]*np.roll(n, -1, 0)
        n = (k[0]*np.roll(n, 1, 1) + k[1]*n + k[2]*np.roll(n, -1, 1)) * 2.0
        lum = out.mean(-1, keepdims=True)
        mask = 4 * lum * (1 - lum)            # strongest in midtones
        out = out + (grain / 255.0) * n[..., None] * mask
    return to_im(out)

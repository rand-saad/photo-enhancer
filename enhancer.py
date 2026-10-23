"""Core image enhancement: Real-ESRGAN x4plus via spandrel, tiled for low memory."""
import os
import urllib.request

import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader

MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RealESRGAN_x4plus.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


def get_model():
    """Load the model once (downloads the weights on first run)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            print("Downloading model weights...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        _model = ModelLoader().load_from_file(MODEL_PATH).to(DEVICE).eval()
        if DEVICE.type == "cuda":
            _model = _model.half()
    return _model


@torch.inference_mode()
def upscale_x4(img: Image.Image, tile: int = 256, pad: int = 16) -> Image.Image:
    """Run the 4x model tile by tile so big photos fit in memory."""
    model = get_model()
    scale = model.scale
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    if DEVICE.type == "cuda":
        x = x.half()
    _, _, h, w = x.shape
    out = torch.zeros((1, 3, h * scale, w * scale), dtype=x.dtype, device=DEVICE)

    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            # padded tile avoids visible seams
            py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
            py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
            res = model(x[:, :, py0:py1, px0:px1])
            oy, ox = (y0 - py0) * scale, (x0 - px0) * scale
            out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = res[
                :, :, oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale
            ]

    res = out.squeeze(0).float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((res * 255.0).round().astype(np.uint8))


def enhance(img: Image.Image, out_scale: int = 2, strength: float = 0.8,
            max_side: int = 1280) -> Image.Image:
    """
    Enhance a photo.
    out_scale: final size relative to the input (1, 2 or 4).
    strength:  0..1, blend between plain resize (0) and full AI result (1).
               Lower values look more natural.
    max_side:  inputs larger than this are shrunk first (speed on CPU).
    """
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        r = max_side / max(w, h)
        img = img.resize((round(w * r), round(h * r)), Image.LANCZOS)
        w, h = img.size

    ai = upscale_x4(img)
    target = (w * out_scale, h * out_scale)
    if ai.size != target:
        ai = ai.resize(target, Image.LANCZOS)

    if strength < 1.0:
        base = img.resize(target, Image.LANCZOS)
        ai = Image.blend(base, ai, strength)
    return ai

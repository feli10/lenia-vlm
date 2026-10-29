"""A frozen pretrained image model (DINOv2 by default) as an embedder: the MAP-Elites form axis and near-duplicate
checks. Moved here verbatim on 2026-10-28 from src/lenia/distill.py (phase 3, now in archive/).
Needs PyTorch (and the internet once, to fetch the weights); model="fake" works without it, for tests.
"""
from __future__ import annotations

import numpy as np

MODELS = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "clip_vitb32": 512}


def _views(img) -> list:
    """A two-view image (width about 2x height) -> [motion view, close-up block]; otherwise [img]."""
    w, h = img.size
    if w > 1.6 * h:
        return [img.crop((0, 0, h, h)), img.crop((w - h, 0, w, h))]
    return [img]


def _load_model(name: str, device: str):
    import torch
    if name.startswith("dinov2"):
        m = torch.hub.load("facebookresearch/dinov2", name).to(device).eval()
        mean, std, size = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
        return (lambda x: m(x)), mean, std, size
    if name == "clip_vitb32":
        import open_clip
        m, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        m = m.to(device).eval()
        mean, std, size = (0.4815, 0.4578, 0.4082), (0.2686, 0.2613, 0.2758), 224
        return (lambda x: m.encode_image(x)), mean, std, size
    raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """A frozen image model, loaded once: embed(list of PIL images) -> (n, dims) array, two-view images
    split into their two views and the vectors joined (as `embed` does for a pool)."""

    def __init__(self, model: str = "dinov2_vits14", device: str | None = None):
        self.model = model
        if model == "fake":                 # tests: no PyTorch; 8x8 grey thumbnails of each view
            self.device = "cpu"
            return
        import torch
        self.device = device or _device()
        self.fwd, mean, std, self.size = _load_model(model, self.device)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(self.device)

    def __call__(self, images: list, batch: int = 64) -> np.ndarray:
        from PIL import Image
        if self.model == "fake":
            return np.array([np.concatenate([np.asarray(v.convert("L").resize((8, 8)), np.float32).ravel() / 255
                                             for v in _views(im)]) for im in images])
        import torch
        out = []
        for k in range(0, len(images), batch):
            views = [_views(im) for im in images[k:k + batch]]
            nv = len(views[0])
            x = np.stack([np.asarray(v.resize((self.size, self.size), Image.BICUBIC), np.float32) / 255 for vs in views for v in vs])
            xt = (torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device) - self.mean) / self.std
            with torch.no_grad():
                e = self.fwd(xt).float().cpu().numpy()
            out.append(e.reshape(len(views), nv * e.shape[-1]))
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)

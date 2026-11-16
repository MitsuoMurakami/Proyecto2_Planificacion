"""Modelos de visión usados para control de calidad y métricas.

Se cargan de forma perezosa. Si `transformers`/`lpips` o la GPU no están
disponibles (modo de prueba sin GPU), se usan aproximaciones deterministas
basadas en NumPy/OpenCV. Cada función devuelve también qué backend se usó
para que las métricas reportadas sean honestas ("dinov2" vs "proxy").
"""
from __future__ import annotations

import functools
import logging

import numpy as np
from PIL import Image

from .config import SCENE_PROMPTS, SETTINGS

log = logging.getLogger(__name__)


def _device() -> str:
    if SETTINGS.device != "auto":
        return SETTINGS.device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Carga perezosa
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _dino():
    if not SETTINGS.use_vision_models:
        return None
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, AutoModel

        proc = AutoImageProcessor.from_pretrained(SETTINGS.dino_id)
        model = AutoModel.from_pretrained(SETTINGS.dino_id).to(_device()).eval()
        return proc, model
    except Exception as exc:  # pragma: no cover - depende del entorno
        log.warning("DINOv2 no disponible (%s); se usa proxy", exc)
        return None


@functools.lru_cache(maxsize=1)
def _clip():
    if not SETTINGS.use_vision_models:
        return None
    try:
        from transformers import CLIPModel, CLIPProcessor

        proc = CLIPProcessor.from_pretrained(SETTINGS.clip_id)
        model = CLIPModel.from_pretrained(SETTINGS.clip_id).to(_device()).eval()
        return proc, model
    except Exception as exc:  # pragma: no cover
        log.warning("CLIP no disponible (%s); se usa proxy", exc)
        return None


@functools.lru_cache(maxsize=1)
def _lpips():
    if not SETTINGS.use_vision_models:
        return None
    try:
        import lpips

        return lpips.LPIPS(net="alex", verbose=False).to(_device()).eval()
    except Exception as exc:  # pragma: no cover
        log.warning("LPIPS no disponible (%s); se usa proxy", exc)
        return None


@functools.lru_cache(maxsize=1)
def _face_cascade():
    """Haar cascade de OpenCV 4.x. OpenCV 5 la movió a contrib: en ese caso devuelve None
    y la detección de personas depende solo de CLIP (se avisa en el log)."""
    import cv2

    if not hasattr(cv2, "CascadeClassifier"):
        log.warning("OpenCV %s sin CascadeClassifier: instala opencv-python-headless<5 para detectar rostros",
                    cv2.__version__)
        return None
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _small_gray(img: Image.Image, size: int = 64) -> np.ndarray:
    return np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0


def _edges(img: Image.Image, size: int = 128) -> np.ndarray:
    import cv2

    g = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR))
    return cv2.Canny(g, 50, 150).astype(np.float32) / 255.0


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def structural_similarity(original: Image.Image, variation: Image.Image) -> tuple[float, str]:
    """Similitud de composición (DINOv2 coseno). Proxy: estructura de bordes + luminancia."""
    m = _dino()
    if m is not None:
        import torch

        proc, model = m
        with torch.no_grad():
            inp = proc(images=[original.convert("RGB"), variation.convert("RGB")], return_tensors="pt").to(_device())
            feats = model(**inp).last_hidden_state[:, 0]
            feats = torch.nn.functional.normalize(feats, dim=-1)
            return float((feats[0] * feats[1]).sum().item()), "dinov2"
    # Proxy: combinación de correlación de bordes (composición) y de luminancia centrada.
    ea, eb = _edges(original), _edges(variation)
    import cv2

    ea = cv2.GaussianBlur(ea, (7, 7), 0)
    eb = cv2.GaussianBlur(eb, (7, 7), 0)
    ga, gb = _small_gray(original), _small_gray(variation)
    ga, gb = ga - ga.mean(), gb - gb.mean()
    score = 0.5 * _cos(ea, eb) + 0.5 * max(0.0, _cos(ga, gb))
    return float(np.clip(score, 0.0, 1.0)), "proxy"


def perceptual_distance(a: Image.Image, b: Image.Image) -> tuple[float, str]:
    """LPIPS entre dos imágenes. Proxy: distancia media en color + bordes normalizada a [0,1]."""
    m = _lpips()
    if m is not None:
        import torch

        def to_t(im):
            arr = np.asarray(im.convert("RGB").resize((256, 256)), dtype=np.float32) / 127.5 - 1.0
            return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(_device())

        with torch.no_grad():
            return float(m(to_t(a), to_t(b)).item()), "lpips"
    ca = np.asarray(a.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
    cb = np.asarray(b.convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
    color = float(np.abs(ca - cb).mean())
    edge = float(np.abs(_edges(a) - _edges(b)).mean())
    return float(np.clip(1.6 * color + 0.8 * edge, 0.0, 1.0)), "proxy"


def mean_pairwise_lpips(images: list[Image.Image]) -> tuple[float, str]:
    if len(images) < 2:
        return 0.0, "n/a"
    vals, backend = [], "proxy"
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            v, backend = perceptual_distance(images[i], images[j])
            vals.append(v)
    return float(np.mean(vals)), backend


def clip_scores(img: Image.Image, texts: list[str]) -> list[float] | None:
    """Similitud coseno imagen-texto con CLIP; None si CLIP no está disponible."""
    m = _clip()
    if m is None:
        return None
    import torch

    proc, model = m
    with torch.no_grad():
        inp = proc(text=texts, images=img.convert("RGB"), return_tensors="pt", padding=True).to(_device())
        out = model(**inp)
        ie = torch.nn.functional.normalize(out.image_embeds, dim=-1)
        te = torch.nn.functional.normalize(out.text_embeds, dim=-1)
        return (ie @ te.T)[0].tolist()


def clip_image_similarity(a: Image.Image, b: Image.Image) -> float | None:
    m = _clip()
    if m is None:
        return None
    import torch

    proc, model = m
    with torch.no_grad():
        inp = proc(images=[a.convert("RGB"), b.convert("RGB")], return_tensors="pt").to(_device())
        e = torch.nn.functional.normalize(model.get_image_features(**inp), dim=-1)
        return float((e[0] * e[1]).sum().item())


def classify_scene(img: Image.Image) -> tuple[str, str]:
    """Clasifica la escena (montaña, costa, bosque, desierto, nieve) con CLIP zero-shot."""
    names = list(SCENE_PROMPTS)
    scores = clip_scores(img, [SCENE_PROMPTS[n] for n in names])
    if scores is not None:
        return names[int(np.argmax(scores))], "clip"
    # Heurística de color para el modo sin GPU.
    arr = np.asarray(img.convert("HSV").resize((64, 64)), dtype=np.float32)
    h, s, v = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
    rgb = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.float32)
    r, g, b = rgb[..., 0].mean(), rgb[..., 1].mean(), rgb[..., 2].mean()
    if v > 170 and s < 50:
        return "nieve", "proxy"
    if b > r and b > g and b - g > 8:
        return "costa", "proxy"
    if g > r and g > b:
        return "bosque", "proxy"
    if r > b + 25 and 15 < h < 45:
        return "desierto", "proxy"
    return "montaña", "proxy"


def detect_faces(img: Image.Image) -> int:
    """Número de rostros detectados (Haar cascade de OpenCV, funciona sin GPU)."""
    cascade = _face_cascade()
    if cascade is None:
        return 0
    g = np.asarray(img.convert("L").resize((512, 512)))
    faces = cascade.detectMultiScale(g, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
    return int(len(faces))


def detect_people_or_text(img: Image.Image, threshold: float) -> dict:
    """Guardrail de salida: rostros (OpenCV) + personas/texto/contenido no apto (CLIP)."""
    result = {"faces": detect_faces(img), "clip": None, "flagged": False, "reasons": []}
    if result["faces"] > 0:
        result["flagged"] = True
        result["reasons"].append("rostro detectado")
    probes = [
        "a photo of a person",
        "an image with written text, letters or a watermark",
        "explicit or violent content",
        "an empty natural landscape",
    ]
    scores = clip_scores(img, probes)
    if scores is not None:
        result["clip"] = dict(zip(["persona", "texto", "no_apto", "paisaje"], scores))
        for key, score in zip(["persona", "texto", "no_apto"], scores[:3]):
            if score > threshold and score > scores[3]:
                result["flagged"] = True
                result["reasons"].append(f"{key} (CLIP {score:.2f})")
    return result

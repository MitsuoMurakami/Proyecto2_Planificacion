"""Modo de prueba sin GPU.

Aplica transformaciones deterministas de color/textura (PIL + NumPy) que imitan
cada variante. No es un modelo generativo: sirve para desarrollar la interfaz,
el registro, el planificador y el agente sin GPU, y para la demo de respaldo.
La semilla introduce variación controlada, igual que en los modelos reales.
"""
from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from ..config import IMAGE_SIZE
from .base import GenRequest, Generator


def _arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _img(a: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def _grade(a: np.ndarray, mul=(1, 1, 1), add=(0, 0, 0), gamma=1.0) -> np.ndarray:
    return np.clip(a ** gamma * np.array(mul) + np.array(add), 0, 1)


def _sky_gradient(a: np.ndarray, top, strength: float) -> np.ndarray:
    h = a.shape[0]
    w = np.linspace(1, 0, h)[:, None, None] ** 1.5 * strength
    return a * (1 - w) + np.array(top) * w


def _grain(a: np.ndarray, rng: np.random.Generator, amount: float) -> np.ndarray:
    return a + rng.normal(0, amount, a.shape[:2])[..., None]


def _vignette(a: np.ndarray, k: float = 0.5) -> np.ndarray:
    h, w = a.shape[:2]
    y, x = np.ogrid[-1:1:h * 1j, -1:1:w * 1j]
    return a * (1 - k * np.clip(x ** 2 + y ** 2, 0, 1))[..., None]


def _stars(a: np.ndarray, rng: np.random.Generator, n: int = 600) -> np.ndarray:
    h, w = a.shape[:2]
    ys = (rng.random(n) ** 2 * h * 0.45).astype(int)
    xs = rng.integers(0, w, n)
    a = a.copy()
    a[ys, xs] = np.maximum(a[ys, xs], rng.uniform(0.6, 1.0, (n, 1)))
    return a


def _fog(a: np.ndarray, density: float) -> np.ndarray:
    h = a.shape[0]
    w = (np.linspace(0.4, 1.0, h)[:, None, None]) * density
    return a * (1 - w) + 0.85 * w


def _rain(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    a = _arr(img)
    streaks = np.zeros(a.shape[:2], np.float32)
    n = 2500
    ys, xs = rng.integers(0, a.shape[0], n), rng.integers(0, a.shape[1], n)
    streaks[ys, xs] = 1.0
    s = Image.fromarray((streaks * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
    s = s.rotate(-12, resample=Image.BILINEAR).filter(ImageFilter.BoxBlur(1))
    s = np.asarray(s, np.float32)[..., None] / 255.0
    return _img(a * 0.85 + s * 0.6)


def _apply(img: Image.Image, axis: str, variant: str, rng: np.random.Generator, prompt: str) -> Image.Image:
    j = lambda s=0.06: 1 + rng.uniform(-s, s)  # jitter dependiente de la semilla  # noqa: E731
    a = _arr(img)
    v = variant
    if axis == "hora":
        if v == "atardecer":
            a = _sky_gradient(_grade(a, (1.15 * j(), 0.92, 0.7), (0.05, 0.0, 0.0)), (1.0, 0.55, 0.2), 0.35 * j())
        elif v == "amanecer":
            a = _sky_gradient(_grade(a, (1.08, 0.95 * j(), 0.95), (0.04, 0.02, 0.05)), (1.0, 0.7, 0.75), 0.3 * j())
        elif v == "hora azul":
            a = _grade(a, (0.6, 0.72, 1.05 * j()), (0.0, 0.02, 0.08), 1.2)
        elif v == "noche estrellada":
            a = _stars(_grade(a, (0.18, 0.22, 0.38 * j()), (0, 0, 0.03), 1.4), rng)
    elif axis == "estilo":
        if v == "óleo impresionista":
            im = _img(a).filter(ImageFilter.ModeFilter(int(5 + rng.integers(0, 3))))
            im = ImageEnhance.Color(im).enhance(1.5 * j())
            return im.filter(ImageFilter.SMOOTH_MORE)
        if v == "acuarela":
            im = _img(a).filter(ImageFilter.MedianFilter(7)).filter(ImageFilter.SMOOTH_MORE)
            im = ImageOps.posterize(im, 5)
            paper = _arr(im) * 0.85 + 0.15 + rng.normal(0, 0.015, a.shape[:2])[..., None]
            return _img(paper)
        if v == "grabado":
            g = ImageOps.grayscale(_img(a)).filter(ImageFilter.FIND_EDGES)
            g = ImageOps.invert(ImageOps.autocontrast(g))
            base = ImageOps.grayscale(_img(a)).point(lambda p: 255 if p > 110 * j() else 40)
            return Image.blend(g, base, 0.35).convert("RGB")
        if v == "pastel":
            im = ImageEnhance.Color(_img(a).filter(ImageFilter.GaussianBlur(2))).enhance(0.7)
            return _img(_arr(im) * 0.75 + 0.25 * j(0.03))
    elif axis == "estacion":
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        green = np.clip((g - np.maximum(r, b)) * 4, 0, 1)[..., None]
        if v == "invierno":
            snow = np.clip((a.mean(-1, keepdims=True) - 0.25) * 2.2 * j(), 0, 1)
            a = _grade(a, (0.85, 0.9, 1.0)) * (1 - snow) + snow * 0.95
        elif v == "otoño":
            a = a * (1 - green) + green * np.array([0.85 * j(), 0.42, 0.1]) * a.mean(-1, keepdims=True) * 2.2
        elif v == "primavera":
            a = _grade(a, (0.95, 1.12 * j(), 0.92), (0.02, 0.03, 0.0))
            n = 900
            ys, xs = rng.integers(a.shape[0] // 2, a.shape[0], n), rng.integers(0, a.shape[1], n)
            a = a.copy()
            a[ys, xs] = np.array([1.0, 0.75, 0.85])
        elif v == "verano seco":
            a = a * (1 - green) + green * np.array([0.8, 0.68 * j(), 0.35]) * a.mean(-1, keepdims=True) * 2
            a = _grade(a, (1.08, 1.02, 0.9), gamma=0.9)
    elif axis == "clima":
        if v == "niebla":
            a = _fog(a, 0.55 * j())
        elif v == "lluvia ligera":
            return _rain(_img(_grade(a, (0.75, 0.78, 0.82 * j()), gamma=1.1)), rng)
        elif v == "tormenta":
            a = _sky_gradient(_grade(a, (0.55, 0.58, 0.65 * j()), gamma=1.25), (0.15, 0.17, 0.22), 0.45)
        elif v == "cielo despejado":
            a = _sky_gradient(_grade(a, (1.0, 1.02, 1.05), gamma=0.95), (0.35, 0.6, 1.0), 0.3 * j())
    elif axis == "epoca":
        if v == "años 70":
            a = _grain(_grade(a, (1.1, 0.98, 0.78), (0.06, 0.04, 0.0), 0.95 * j(0.03)), rng, 0.03)
        elif v == "años 20 (B/N)":
            gray = a.mean(-1, keepdims=True).repeat(3, -1)
            a = _vignette(_grain(_grade(gray, (1, 0.97, 0.9), gamma=1.1), rng, 0.05), 0.6 * j())
        elif v == "años 50":
            a = np.asarray(ImageEnhance.Color(_img(a)).enhance(1.6 * j()), np.float32) / 255.0
            a = _grade(a, (1.05, 1.0, 0.92), (0.0, 0.02, 0.03))
        elif v == "años 90":
            a = _vignette(_grain(_grade(a, (1.05, 1.07, 0.95), (0.04, 0.04, 0.02), 0.85), rng, 0.025), 0.35 * j())
    else:  # personalizada: tono derivado del texto de la instrucción
        h = int(hashlib.sha256(prompt.encode()).hexdigest(), 16)
        tint = np.array([0.8 + (h % 40) / 100, 0.8 + (h // 40 % 40) / 100, 0.8 + (h // 1600 % 40) / 100])
        a = _grade(a, tint * j(0.04), gamma=0.9 + (h % 20) / 100)
    return _img(a)


class MockGenerator(Generator):
    name = "mock"

    def generate(self, image: Image.Image, req: GenRequest) -> Image.Image:
        rng = np.random.default_rng(req.seed)
        out = _apply(image.convert("RGB"), req.axis, req.variant, rng, req.prompt)
        return out.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    def params(self) -> dict:
        return {"method": self.name, "note": "transformaciones deterministas sin GPU"}

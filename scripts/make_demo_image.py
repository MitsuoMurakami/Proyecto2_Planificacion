"""Genera un paisaje sintético (montañas + lago + bosque) para probar la app sin fotos propias.

    python scripts/make_demo_image.py examples/paisaje_demo.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def synthetic_landscape(size: int = 1024, seed: int = 3) -> Image.Image:
    rng = np.random.default_rng(seed)
    h = w = size
    y = np.linspace(0, 1, h)[:, None]
    sky = np.concatenate([0.35 + 0.4 * y, 0.55 + 0.3 * y, 0.85 + 0.1 * y], axis=1)
    arr = np.repeat(sky[:, None, :], w, axis=1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    # Montañas (dos capas) con ruido en la cresta.
    for base, amp, color in [(0.45, 0.22, (95, 105, 125)), (0.55, 0.15, (60, 80, 70))]:
        xs = np.arange(0, w + 8, 8)
        ridge = base * h - amp * h * np.abs(np.sin(xs / w * np.pi * rng.uniform(1.5, 3) + rng.uniform(0, 3)))
        ridge += rng.normal(0, 6, xs.shape).cumsum() * 0.3
        d.polygon([(0, h)] + list(zip(xs.tolist(), ridge.tolist())) + [(w, h)], fill=color)
    # Nieve en las cumbres.
    d.polygon([(w * 0.28, h * 0.27), (w * 0.33, h * 0.33), (w * 0.23, h * 0.33)], fill=(240, 240, 245))
    # Lago y orilla.
    d.rectangle([0, int(h * 0.68), w, h], fill=(70, 110, 150))
    for _ in range(140):  # árboles
        x = rng.integers(0, w)
        yb = int(h * rng.uniform(0.62, 0.7))
        s = rng.integers(14, 34)
        d.polygon([(x, yb - 2.5 * s), (x - s / 2, yb), (x + s / 2, yb)], fill=(25, int(rng.integers(60, 95)), 40))
    for _ in range(400):  # reflejos
        x, yy = rng.integers(0, w), rng.integers(int(h * 0.7), h)
        d.line([(x, yy), (x + rng.integers(10, 40), yy)], fill=(150, 180, 210), width=1)
    noise = rng.normal(0, 6, (h, w, 1))
    out = np.clip(np.asarray(img, np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out).filter(ImageFilter.SMOOTH)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "examples/paisaje_demo.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    synthetic_landscape().save(out, quality=92)
    print(f"Guardado en {out}")

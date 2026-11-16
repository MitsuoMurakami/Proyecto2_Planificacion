"""Paso 1 del flujo: preprocesamiento de la foto del usuario.

- elimina metadatos EXIF (incluida la ubicación GPS) reconstruyendo la imagen
  solo a partir de sus píxeles;
- normaliza a 1024×1024 (recorte central + redimensionamiento);
- descarta imágenes borrosas, con rostros o con marcas de agua;
- clasifica la escena con CLIP (montaña, costa, bosque, desierto, nieve);
- calcula un hash SHA-256: el registro guarda el hash, nunca la foto.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

from . import vision
from .config import IMAGE_SIZE, SETTINGS


@dataclass
class PreprocessResult:
    ok: bool
    image: Image.Image | None
    image_hash: str
    scene: str | None = None
    scene_backend: str | None = None
    blur_score: float = 0.0
    faces: int = 0
    watermark: bool = False
    had_exif: bool = False
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.ok:
            return "Imagen rechazada: " + "; ".join(self.reasons)
        exif = "EXIF eliminado" if self.had_exif else "sin EXIF"
        return f"Escena: {self.scene} ({self.scene_backend}) · nitidez {self.blur_score:.0f} · {exif}"


def strip_exif(img: Image.Image) -> tuple[Image.Image, bool]:
    """Devuelve una copia sin metadatos. Respeta la orientación EXIF antes de borrarla."""
    had_exif = bool(img.info.get("exif")) or bool(getattr(img, "getexif", lambda: {})())
    img = ImageOps.exif_transpose(img)
    rgb = img.convert("RGB")
    clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())  # solo píxeles: sin EXIF/ICC/XMP
    return clean, had_exif


def center_square(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    return ImageOps.fit(img, (size, size), method=Image.LANCZOS, centering=(0.5, 0.5))


def blur_score(img: Image.Image) -> float:
    """Varianza del laplaciano (mayor = más nítida), calculada a 512 px."""
    import cv2

    g = np.asarray(img.convert("L").resize((512, 512)))
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def has_watermark(img: Image.Image) -> bool:
    """Detección conservadora de marcas de agua.

    Con CLIP: sonda de texto/marca de agua. Sin CLIP: busca en las esquinas
    regiones con trazos finos de alto contraste muy por encima de la textura
    típica del resto de la imagen.
    """
    scores = vision.clip_scores(img, ["a photo with a watermark or copyright text overlay", "a clean photo with no text"])
    if scores is not None:
        return scores[0] > scores[1] + 0.03
    import cv2

    g = np.asarray(img.convert("L").resize((512, 512)))
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, np.ones((9, 9), np.uint8))
    blackhat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, np.ones((9, 9), np.uint8))
    strokes = ((tophat > 60) | (blackhat > 60)).astype(np.float32)
    h, w = strokes.shape
    b = 64
    corners = [strokes[:b, :b * 2], strokes[:b, -b * 2:], strokes[-b:, :b * 2], strokes[-b:, -b * 2:]]
    center = strokes[b:-b, b:-b].mean() + 1e-3
    return any(c.mean() > 0.08 and c.mean() > 6 * center for c in corners)


def image_hash(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def preprocess(img: Image.Image, check_faces: bool = True) -> PreprocessResult:
    clean, had_exif = strip_exif(img)
    square = center_square(clean)
    h = image_hash(square)
    res = PreprocessResult(ok=True, image=square, image_hash=h, had_exif=had_exif)

    res.blur_score = blur_score(square)
    if res.blur_score < SETTINGS.thresholds.blur_laplacian_min:
        res.reasons.append(f"imagen borrosa (nitidez {res.blur_score:.0f} < {SETTINGS.thresholds.blur_laplacian_min:.0f})")
    if check_faces:
        res.faces = vision.detect_faces(square)
        if res.faces:
            res.reasons.append(f"se detectaron {res.faces} rostro(s); solo se aceptan paisajes sin personas")
    res.watermark = has_watermark(square)
    if res.watermark:
        res.reasons.append("posible marca de agua")

    if res.reasons:
        res.ok = False
        res.image = None
        return res
    res.scene, res.scene_backend = vision.classify_scene(square)
    return res

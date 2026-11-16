"""Pasos 2-3 del flujo: generación con control de calidad y reintento.

Por cada variación:
- se mide la similitud DINOv2 con el original; si es baja, se reintenta con
  otra semilla y, si persiste, se conserva la mejor y se muestra un aviso;
- se revisa la salida (rostros + CLIP); si inserta personas/texto/contenido no
  apto se reintenta y, si persiste, la imagen se descarta;
- si la GPU se queda sin memoria, se activa la descarga a CPU y se reintenta.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

from PIL import Image

from . import vision
from .config import SETTINGS, prompt_for
from .generation import GenRequest, Generator

log = logging.getLogger(__name__)


@dataclass
class QCResult:
    image: Image.Image | None
    seed: int
    attempts: int
    dino: float | None
    dino_backend: str
    clip_adherence: float | None = None
    warning: str | None = None
    discarded: bool = False
    flags: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower()


def _generate_once(gen: Generator, image: Image.Image, req: GenRequest) -> Image.Image:
    try:
        return gen.generate(image, req)
    except RuntimeError as exc:
        pipe = getattr(gen, "pipe", None)
        if _is_oom(exc) and pipe is not None and hasattr(pipe, "enable_model_cpu_offload"):
            log.warning("GPU sin memoria: se activa la descarga a CPU y se reintenta")
            import torch

            torch.cuda.empty_cache()
            pipe.enable_model_cpu_offload()
            return gen.generate(image, req)
        raise


def generate_with_qc(gen: Generator, original: Image.Image, axis: str, variant: str, seed: int,
                     custom_text: str | None = None, max_retries: int | None = None) -> QCResult:
    th = SETTINGS.thresholds
    max_retries = th.max_retries if max_retries is None else max_retries
    prompt = prompt_for(axis, variant, custom_text)
    rng = random.Random(seed)
    t0 = time.perf_counter()
    best: QCResult | None = None
    flags_all: list[str] = []
    for attempt in range(1, max_retries + 2):
        s = seed if attempt == 1 else rng.randint(0, 2**31 - 1)
        img = _generate_once(gen, original, GenRequest(axis, variant, prompt, s))
        dino, backend = vision.structural_similarity(original, img)
        safety = vision.detect_people_or_text(img, th.clip_text_max)
        cand = QCResult(img, s, attempt, dino, backend)
        if safety["flagged"]:
            flags_all += safety["reasons"]
            continue  # salida no apta: nunca se conserva
        if best is None or dino > (best.dino or -1):
            best = cand
        if dino >= th.dino_min:
            break
    elapsed = time.perf_counter() - t0
    if best is None:
        return QCResult(None, seed, attempt, None, "n/a", discarded=True, flags=flags_all, seconds=elapsed,
                        warning="Descartada: la salida insertaba " + ", ".join(sorted(set(flags_all))))
    best.attempts = attempt
    best.flags = flags_all
    best.seconds = elapsed
    if best.dino is not None and best.dino < th.dino_min:
        best.warning = (f"Aviso: la composición cambió más de lo esperado (similitud {best.dino:.2f} < "
                        f"{th.dino_min:.2f}) tras {attempt} intentos.")
    best.clip_adherence = vision.clip_image_similarity(original, best.image)
    return best

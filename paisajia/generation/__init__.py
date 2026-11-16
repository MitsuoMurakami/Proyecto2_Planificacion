"""Fábrica de generadores: `flux` (principal), `sd` (línea base) y `mock` (sin GPU)."""
from __future__ import annotations

import functools
import logging

from .base import GenRequest, Generator
from .mock import MockGenerator

log = logging.getLogger(__name__)

__all__ = ["GenRequest", "Generator", "get_generator", "MockGenerator"]


@functools.lru_cache(maxsize=3)
def get_generator(name: str = "mock", fallback_to_mock: bool = True) -> Generator:
    name = name.lower()
    try:
        if name in ("flux", "flux2", "flux2-klein"):
            from .flux import FluxKleinGenerator

            return FluxKleinGenerator()
        if name in ("sd", "sd15", "controlnet"):
            from .sd_controlnet import SDControlNetGenerator

            return SDControlNetGenerator()
        if name == "mock":
            return MockGenerator()
        raise ValueError(f"Generador desconocido: {name}")
    except (ImportError, RuntimeError, OSError) as exc:
        if not fallback_to_mock:
            raise
        log.warning("No se pudo cargar '%s' (%s). Se usa el modo de prueba sin GPU.", name, exc)
        return MockGenerator()

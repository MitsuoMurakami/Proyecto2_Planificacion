"""Paso 6 del flujo: agente LLM + verificador + plantilla de respaldo.

Intento inicial + hasta dos reintentos indicando los errores al LLM. Si sigue
fallando, se usa la plantilla determinista y el evento se registra como fallo
del LLM.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import SETTINGS
from .llm import BaseLLM, LLMError
from .template import build_template
from .verifier import VerificationResult, verify


@dataclass
class Explanation:
    output: dict
    attempts: int
    first_try_valid: bool
    used_template: bool
    llm_name: str
    errors: list[list[str]] = field(default_factory=list)  # errores por intento
    first_raw: str | None = None                             # salida sin verificar (para medir alucinación)
    seconds: float = 0.0

    @property
    def source(self) -> str:
        return "plantilla determinista (fallo del LLM)" if self.used_template else f"agente LLM ({self.llm_name})"


def explain(log: list[dict], package_comments: dict[int, str], llm: BaseLLM,
            max_retries: int | None = None) -> Explanation:
    if any(r.get("decision") is None for r in log):
        raise ValueError("El resumen solo se habilita cuando todas las imágenes tienen decisión.")
    max_retries = SETTINGS.thresholds.llm_max_retries if max_retries is None else max_retries
    t0 = time.perf_counter()
    llm.reset()
    errors: list[list[str]] = []
    feedback: list[str] = []
    first_raw = None
    for attempt in range(1, max_retries + 2):
        try:
            raw = llm.summarize(log, package_comments, feedback)
        except LLMError as exc:
            errors.append([f"LLM no disponible: {exc}"])
            break  # un error de API no se corrige reintentando con feedback
        if first_raw is None:
            first_raw = raw
        res: VerificationResult = verify(raw, log, package_comments)
        if res.valid:
            return Explanation(res.parsed, attempt, attempt == 1, False, llm.name, errors, first_raw,
                               time.perf_counter() - t0)
        errors.append(res.errors)
        feedback = res.errors
    out = build_template(log, package_comments)
    return Explanation(out, len(errors), False, True, llm.name, errors, first_raw, time.perf_counter() - t0)

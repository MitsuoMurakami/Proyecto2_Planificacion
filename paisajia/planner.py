"""Paso 7 del flujo: planificación de rondas con un bandido multibrazo.

Cada brazo es un par (eje, variante) con un posterior Beta(1 + aceptadas,
1 + rechazadas) calculado del registro de la sesión.

Regla por posición de la ronda anterior:
- aceptada  -> explotación: se repite la misma variante con otra semilla;
- rechazada -> exploración: muestreo de Thompson entre las demás variantes
  del mismo eje (las no probadas tienen prior optimista frente a las rechazadas).

La tabla de planificación se calcula del registro, no la redacta el LLM, por lo
que no puede alucinar. Cada imagen nueva lleva el motivo de su elección.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .config import AXES, AXIS_NAMES, CUSTOM_AXIS, N_VARIATIONS, base_variant


@dataclass
class Slot:
    axis: str
    variant: str
    seed: int
    reason: str
    custom_text: str | None = None
    theta: float | None = None  # muestra de Thompson usada (si aplica)


@dataclass
class PlanRow:
    """Fila de la tabla de planificación mostrada al usuario."""
    posicion: int
    anterior: str
    decision: str
    accion: str
    siguiente: str
    motivo: str
    probabilidades: dict = field(default_factory=dict)


def _seed(rng: random.Random) -> int:
    return rng.randint(0, 2**31 - 1)


def initial_round(axes: list[str] | None = None, chosen_variants: dict[str, str] | None = None,
                  custom_text: str | None = None, seed: int | None = None) -> list[Slot]:
    """Ronda 1: siempre exactamente cinco imágenes.

    - paquete completo (axes=None): un slot por eje con su variante base;
    - ejes individuales: se reparten las cinco posiciones entre los ejes elegidos,
      empezando por la variante elegida por el usuario y siguiendo con las demás;
    - instrucción propia: ocupa una posición.
    """
    rng = random.Random(seed)
    axes = list(AXIS_NAMES) if axes is None else list(axes)
    chosen_variants = chosen_variants or {}
    slots: list[Slot] = []
    if custom_text:
        slots.append(Slot(CUSTOM_AXIS, "instrucción propia", _seed(rng), "instrucción escrita por el usuario", custom_text))
    if not axes and not slots:
        axes = list(AXIS_NAMES)
    queues = {}
    for a in axes:
        variants = list(AXES[a]["variants"])
        first = chosen_variants.get(a) or variants[0]
        queues[a] = [first] + [v for v in variants if v != first]
    i = 0
    while len(slots) < N_VARIATIONS and axes:
        a = axes[i % len(axes)]
        q = queues[a]
        v = q[(i // len(axes)) % len(q)]
        reason = "elegida por el usuario" if chosen_variants.get(a) == v else (
            "variante base del eje" if v == base_variant(a) else "variante adicional del eje elegido")
        slots.append(Slot(a, v, _seed(rng), f"{v}: {reason}"))
        i += 1
    while len(slots) < N_VARIATIONS:  # solo instrucción propia: se repite con otras semillas
        slots.append(Slot(CUSTOM_AXIS, "instrucción propia", _seed(rng), "instrucción propia con otra semilla", custom_text))
    return slots[:N_VARIATIONS]


class ThompsonPlanner:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def posterior(self, stats: dict, axis: str, variant: str) -> tuple[int, int]:
        a, r = stats.get((axis, variant), (0, 0))
        return 1 + a, 1 + r

    def sample(self, stats: dict, axis: str, variant: str) -> float:
        a, b = self.posterior(stats, axis, variant)
        return float(self.np_rng.beta(a, b))

    def next_round(self, previous: list[dict], stats: dict) -> tuple[list[Slot], list[PlanRow]]:
        """`previous`: variaciones activas de la ronda anterior con su decisión.

        Cada elemento: {slot, axis, variant, decision, prompt, custom_text}.
        """
        slots: list[Slot] = []
        table: list[PlanRow] = []
        used: set[tuple[str, str]] = set()
        for prev in sorted(previous, key=lambda p: p["slot"]):
            axis, variant, decision = prev["axis"], prev["variant"], prev["decision"]
            if decision == "aceptada":
                s = Slot(axis, variant, _seed(self.rng), f"{variant}: aceptada, se repite con otra semilla",
                         prev.get("custom_text"))
                accion = "explotar"
                probs = {}
            else:
                s, probs = self._explore(axis, variant, stats, used)
                accion = "explorar"
            used.add((s.axis, s.variant))
            slots.append(s)
            table.append(PlanRow(prev["slot"], f"{AXES.get(axis, {}).get('label', 'Personalizada')}: {variant}",
                                 decision or "—", accion, f"{s.variant}", s.reason, probs))
        return slots, table

    def _explore(self, axis: str, rejected: str, stats: dict, used: set) -> tuple[Slot, dict]:
        if axis == CUSTOM_AXIS:
            # La instrucción propia rechazada se sustituye por el mejor brazo de cualquier eje.
            candidates = [(a, v) for a in AXIS_NAMES for v in AXES[a]["variants"] if (a, v) not in used]
            label = "instrucción propia"
        else:
            candidates = [(axis, v) for v in AXES[axis]["variants"] if v != rejected and (axis, v) not in used]
            if not candidates:  # todas usadas en esta ronda: permitimos repetir
                candidates = [(axis, v) for v in AXES[axis]["variants"] if v != rejected]
            label = rejected
        draws = {(a, v): self.sample(stats, a, v) for a, v in candidates}
        (a, v), theta = max(draws.items(), key=lambda kv: kv[1])
        probs = {f"{va}": round(t, 3) for (ax, va), t in sorted(draws.items(), key=lambda kv: -kv[1])}
        return Slot(a, v, _seed(self.rng), f"{v}: alternativa a {label}, rechazada", theta=theta), probs

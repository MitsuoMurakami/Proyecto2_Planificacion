"""Plantilla determinista de respaldo (si el LLM falla dos veces tras reintentos).

Se construye solo con el registro, por lo que siempre pasa el verificador.
"""
from __future__ import annotations

from collections import defaultdict

from ..config import AXES, CUSTOM_AXIS


def axis_label(axis: str) -> str:
    return AXES[axis]["label"] if axis in AXES else "Instrucción propia"


def _fmt(rows: list[dict]) -> str:
    return ", ".join(f"{r['id']} ({axis_label(r['eje']).lower()}: {r['variante']})" for r in rows) or "ninguna"


def build_template(log: list[dict], package_comments: dict[int, str] | None = None) -> dict:
    package_comments = package_comments or {}
    acc = [r for r in log if r.get("decision") == "aceptada"]
    rej = [r for r in log if r.get("decision") == "rechazada"]
    rounds = sorted({r["ronda"] for r in log})
    parts: list[str] = []
    for rnd in rounds:
        ra = [r for r in acc if r["ronda"] == rnd]
        rr = [r for r in rej if r["ronda"] == rnd]
        parts.append(f"Ronda {rnd}: {len(ra)} aceptada(s) [{_fmt(ra)}] y {len(rr)} rechazada(s) [{_fmt(rr)}].")
    motives = [f"{r['id']}: «{r['motivo']}»" for r in log if r.get("motivo")]
    if motives:
        parts.append("Motivos escritos por el usuario: " + "; ".join(motives) + ".")
    for rnd, c in sorted(package_comments.items()):
        parts.append(f"Comentario del paquete en la ronda {rnd}: «{c}».")
    # Patrón por eje (conteos, sin interpretación).
    per_axis: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in log:
        if r.get("decision") == "aceptada":
            per_axis[r["eje"]][0] += 1
        elif r.get("decision") == "rechazada":
            per_axis[r["eje"]][1] += 1
    if per_axis:
        stats = "; ".join(
            f"{axis_label(a)}: {n_a} de {n_a + n_r}" for a, (n_a, n_r) in sorted(per_axis.items(), key=lambda kv: -kv[1][0])
        )
        parts.append(f"Proporción de aceptación por eje a lo largo de las rondas: {stats}.")
    if len(rounds) > 1:
        first = [r for r in log if r["ronda"] == rounds[0] and r.get("decision")]
        later = [r for r in log if r["ronda"] != rounds[0] and r.get("decision")]
        rate = lambda rows: sum(r["decision"] == "aceptada" for r in rows) / max(1, len(rows))  # noqa: E731
        parts.append(f"Tasa de aceptación: {rate(first):.0%} en la ronda {rounds[0]} y {rate(later):.0%} en las rondas posteriores.")
    return {
        "aceptadas": [r["id"] for r in acc],
        "rechazadas": [r["id"] for r in rej],
        "resumen": " ".join(parts),
    }


__all__ = ["build_template", "axis_label", "CUSTOM_AXIS"]

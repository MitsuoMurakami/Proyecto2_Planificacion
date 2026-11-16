"""Evaluación del ciclo de iteración (O2) con usuarios simulados.

Cada usuario simulado tiene una preferencia oculta p(aceptar | eje, variante).
Comparamos la tasa de aceptación en la ronda 1 frente a las rondas ≥ 2 para:
- Thompson (PaisajIA): explotar lo aceptado, explorar con muestreo de Thompson lo rechazado;
- Aleatorio: cada ronda elige variantes al azar en los mismos ejes (línea base).

No genera imágenes: aísla el efecto de la planificación.

    python scripts/eval_planner.py --users 200 --rounds 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paisajia.config import AXES, AXIS_NAMES  # noqa: E402
from paisajia.planner import Slot, ThompsonPlanner, initial_round  # noqa: E402


def make_user(rng: random.Random) -> dict:
    """Cada usuario tiene 1-2 variantes favoritas por eje (p alta) y el resto con p baja."""
    prefs = {}
    for a in AXIS_NAMES:
        vs = list(AXES[a]["variants"])
        fav = set(rng.sample(vs, rng.choice([1, 2])))
        for v in vs:
            prefs[(a, v)] = rng.uniform(0.65, 0.9) if v in fav else rng.uniform(0.05, 0.3)
    return prefs


def run(policy: str, users: int, rounds: int, seed: int) -> dict:
    rng = random.Random(seed)
    per_round = {r: [] for r in range(1, rounds + 1)}
    for u in range(users):
        prefs = make_user(rng)
        planner = ThompsonPlanner(seed + u)
        stats: dict = {}
        slots = initial_round(seed=seed + u)
        for rnd in range(1, rounds + 1):
            prev = []
            for i, s in enumerate(slots, 1):
                dec = "aceptada" if rng.random() < prefs[(s.axis, s.variant)] else "rechazada"
                a, r = stats.get((s.axis, s.variant), (0, 0))
                stats[(s.axis, s.variant)] = (a + (dec == "aceptada"), r + (dec == "rechazada"))
                per_round[rnd].append(dec == "aceptada")
                prev.append({"slot": i, "axis": s.axis, "variant": s.variant, "decision": dec})
            if policy == "thompson":
                slots, _ = planner.next_round(prev, stats)
            else:
                slots = [Slot(p["axis"], rng.choice(list(AXES[p["axis"]]["variants"])), 0, "aleatorio") for p in prev]
    rates = {r: sum(v) / len(v) for r, v in per_round.items()}
    later = [x for r in range(2, rounds + 1) for x in per_round[r]]
    return {"por_ronda": rates, "ronda_1": rates[1], "rondas_2+": sum(later) / len(later)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="results/eval_planner.json")
    args = ap.parse_args()
    report = {p: run(p, args.users, args.rounds, args.seed) for p in ("thompson", "aleatorio")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Evaluación del agente: exactitud de listas y tasa de alucinación, con y sin verificador.

Genera N sesiones simuladas (por defecto 100) con registros aleatorios (1-4 rondas,
decisiones y motivos aleatorios) y ejecuta el agente sobre cada una.

- "sin verificador": se evalúa la primera salida cruda del LLM.
- "con verificador": se evalúa la salida final (reintentos ×2 + plantilla).

Uso:
    python scripts/eval_agent.py                       # LLM simulado (error_rate=0.3), sin coste
    python scripts/eval_agent.py --llm claude --n 100  # Claude real (requiere ANTHROPIC_API_KEY)
    python scripts/eval_agent.py --llm ollama          # Qwen 2.5 3B local (ollama pull qwen2.5:3b)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paisajia.agent import explain, get_llm, verify  # noqa: E402
from paisajia.agent.verifier import expected_lists  # noqa: E402
from paisajia.config import AXES, AXIS_NAMES, CUSTOM_AXIS  # noqa: E402

MOTIVES = [
    "me encanta la luz", "muy artificial", "demasiado oscuro", "perdió las montañas", "colores bonitos",
    "parece una postal", "no se parece a mi foto", "justo lo que buscaba", "muy saturado", "la niebla tapa todo",
    "me gusta el grano", "prefiero algo más natural",
]
CUSTOM = ["luz de luna llena reflejada en el agua", "colores de película de Wes", "atmósfera de sueño"]


def random_log(rng: random.Random) -> tuple[list[dict], dict[int, str]]:
    rows, comments = [], {}
    n_rounds = rng.randint(1, 4)
    for rnd in range(1, n_rounds + 1):
        for slot in range(1, 6):
            if rng.random() < 0.08:
                axis, variant, instr = CUSTOM_AXIS, "instrucción propia", rng.choice(CUSTOM)
            else:
                axis = rng.choice(AXIS_NAMES)
                variant, instr = rng.choice(list(AXES[axis]["variants"])), None
            dec = rng.choice(["aceptada", "rechazada"])
            rows.append({"id": f"R{rnd}-{slot}", "ronda": rnd, "eje": axis, "variante": variant, "instruccion": instr,
                         "decision": dec, "motivo": rng.choice(MOTIVES) if rng.random() < 0.35 else None,
                         "via": "individual"})
        if rng.random() < 0.2:
            comments[rnd] = rng.choice(["en general bien", "ninguna me convenció del todo", "sigue así"])
    return rows, comments


def lists_exact(out: dict | None, log: list[dict]) -> bool:
    if not isinstance(out, dict):
        return False
    acc, rej = expected_lists(log)
    return sorted(out.get("aceptadas") or []) == sorted(acc) and sorted(out.get("rechazadas") or []) == sorted(rej)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--llm", default="simulado", choices=["simulado", "claude", "ollama"])
    ap.add_argument("--model", default=None, help="modelo de Claude u Ollama (p. ej. qwen2.5:3b)")
    ap.add_argument("--error-rate", type=float, default=0.3, help="solo LLM simulado")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="por defecto results/eval_agent_<llm>.json")
    args = ap.parse_args()
    args.out = args.out or f"results/eval_agent_{args.llm}.json"

    rng = random.Random(args.seed)
    if args.llm == "simulado":
        kw = {"error_rate": args.error_rate, "seed": args.seed}
    else:
        kw = {"model": args.model} if args.model else {}
    llm = get_llm(args.llm, **kw)
    stats = {"raw_valid_json": 0, "raw_lists_exact": 0, "raw_hallucination": 0, "final_lists_exact": 0,
             "final_valid": 0, "first_try_valid": 0, "used_template": 0, "attempts": 0, "seconds": 0.0}
    error_kinds: dict[str, int] = {}
    t0 = time.time()
    for i in range(args.n):
        log, comments = random_log(rng)
        exp = explain(log, comments, llm)
        if exp.first_raw is None and exp.errors and exp.errors[0][0].startswith("LLM no disponible"):
            # Un fallo de conexión no es un error del modelo: no se cuenta como alucinación.
            sys.exit(f"\n{exp.errors[0][0]}")
        try:
            raw = json.loads(exp.first_raw) if exp.first_raw else None
            stats["raw_valid_json"] += isinstance(raw, dict)
        except json.JSONDecodeError:
            raw = None
        stats["raw_lists_exact"] += lists_exact(raw, log)
        raw_check = verify(exp.first_raw or "", log, comments)
        stats["raw_hallucination"] += not raw_check.valid
        for e in raw_check.errors:
            k = e.split(":")[0].split("«")[0].strip()[:50]
            error_kinds[k] = error_kinds.get(k, 0) + 1
        stats["final_lists_exact"] += lists_exact(exp.output, log)
        stats["final_valid"] += verify(exp.output, log, comments).valid
        stats["first_try_valid"] += exp.first_try_valid
        stats["used_template"] += exp.used_template
        stats["attempts"] += exp.attempts
        stats["seconds"] += exp.seconds
        print(f"\r{i + 1}/{args.n}", end="", flush=True)
    n = args.n
    report = {
        "llm": llm.name, "n_sesiones": n, "error_rate_simulado": args.error_rate if args.llm == "simulado" else None,
        "sin_verificador": {
            "json_valido": stats["raw_valid_json"] / n,
            "exactitud_listas": stats["raw_lists_exact"] / n,
            "tasa_alucinacion_o_error": stats["raw_hallucination"] / n,
        },
        "con_verificador": {
            "exactitud_listas": stats["final_lists_exact"] / n,
            "salida_valida": stats["final_valid"] / n,
            "valido_al_primer_intento": stats["first_try_valid"] / n,
            "uso_plantilla": stats["used_template"] / n,
            "intentos_medios": stats["attempts"] / n,
        },
        "segundos_medios": stats["seconds"] / n,
        "errores_detectados_sin_verificador": dict(sorted(error_kinds.items(), key=lambda kv: -kv[1])),
        "duracion_total_s": time.time() - t0,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

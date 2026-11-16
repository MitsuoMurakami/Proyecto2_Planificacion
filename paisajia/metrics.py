"""Métricas de monitoreo calculadas a partir de las sesiones reales (pestaña Monitoreo)."""
from __future__ import annotations

import json

from .config import AXES, SETTINGS
from .registry import Registry

SUS_ITEMS = [
    "Creo que me gustaría usar este sistema con frecuencia.",
    "Encontré el sistema innecesariamente complejo.",
    "Pensé que el sistema era fácil de usar.",
    "Creo que necesitaría el apoyo de un técnico para poder usar este sistema.",
    "Encontré que las diversas funciones del sistema estaban bien integradas.",
    "Pensé que había demasiada inconsistencia en el sistema.",
    "Imagino que la mayoría de las personas aprendería a usar este sistema muy rápidamente.",
    "Encontré el sistema muy engorroso de usar.",
    "Me sentí muy seguro/a usando el sistema.",
    "Necesité aprender muchas cosas antes de poder empezar con este sistema.",
]


def sus_score(answers: list[int]) -> float:
    """Puntaje SUS (0-100) a partir de 10 respuestas Likert 1-5."""
    if len(answers) != 10 or not all(1 <= a <= 5 for a in answers):
        raise ValueError("SUS requiere 10 respuestas entre 1 y 5")
    total = sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))
    return total * 2.5


def _rate(rows: list[dict]) -> float | None:
    return sum(r["decision"] == "aceptada" for r in rows) / len(rows) if rows else None


def summary(reg: Registry, include_simulated: bool = False) -> dict:
    sim = "" if include_simulated else "AND s.simulated=0"
    th = SETTINGS.thresholds
    dec = reg.query(f"""SELECT d.*, v.axis, v.variant FROM decisions d
        JOIN variations v ON v.session_id=d.session_id AND v.var_id=d.var_id AND v.active=1
        JOIN sessions s ON s.session_id=d.session_id WHERE 1=1 {sim}""")
    var = reg.query(f"""SELECT v.* FROM variations v JOIN sessions s ON s.session_id=v.session_id
        WHERE v.active=1 {sim}""")
    rounds = reg.query(f"SELECT r.* FROM rounds r JOIN sessions s ON s.session_id=r.session_id WHERE 1=1 {sim}")
    runs = reg.query(f"SELECT a.* FROM agent_runs a JOIN sessions s ON s.session_id=a.session_id WHERE 1=1 {sim}")
    guard = reg.query("SELECT kind, category, COUNT(*) AS n FROM guardrail_events GROUP BY kind, category")
    surveys = reg.query("SELECT * FROM surveys")
    n_sessions = reg.query(f"SELECT COUNT(*) AS n FROM sessions s WHERE 1=1 {sim}")[0]["n"]

    dinos = [v["dino"] for v in var if v["dino"] is not None]
    gen_s = [r["gen_seconds"] for r in rounds if r["gen_seconds"] is not None]
    lp = [r["lpips_mean"] for r in rounds if r["lpips_mean"] is not None]
    dec_times = [d["decided_at"] - d["shown_at"] for d in dec if d["shown_at"] and d["via"] == "individual"]
    r1 = [d for d in dec if d["round"] == 1]
    r2 = [d for d in dec if d["round"] >= 2]
    per_axis = {}
    for a in list(AXES) + ["personalizada"]:
        rows = [d for d in dec if d["axis"] == a]
        if rows:
            per_axis[AXES.get(a, {}).get("label", "Personalizada")] = {"n": len(rows), "aceptación": _rate(rows)}
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    return {
        "sesiones": n_sessions,
        "rondas": len(rounds),
        "variaciones": len(var),
        "decisiones": len(dec),
        "composicion": {
            "similitud_media": mean(dinos),
            "pct_sobre_umbral": mean([d >= th.dino_min for d in dinos]),
            "backend": sorted({v["dino_backend"] for v in var if v["dino_backend"]}),
            "meta": f"≥ {th.dino_min} en el 80 % de las variaciones",
            "reintentos_medios": mean([v["attempts"] - 1 for v in var]),
            "avisos": sum(1 for v in var if v["warning"]),
        },
        "diversidad": {"lpips_medio": mean(lp), "meta": f"≥ {th.lpips_target}",
                       "backend": sorted({r["lpips_backend"] for r in rounds if r["lpips_backend"]})},
        "tiempo": {"segundos_por_ronda": mean(gen_s), "max": max(gen_s) if gen_s else None,
                   "meta": f"< {th.round_seconds:.0f} s"},
        "iteracion": {"aceptacion_ronda1": _rate(r1), "aceptacion_rondas_2+": _rate(r2), "n_r1": len(r1), "n_r2+": len(r2)},
        "agente": {
            "resumenes": len(runs),
            "validos_al_primer_intento": mean([r["first_try_valid"] for r in runs]),
            "intentos_medios": mean([r["attempts"] for r in runs]),
            "uso_de_plantilla": mean([r["used_template"] for r in runs]),
            "coincidencia_final_con_registro": 1.0 if runs else None,  # garantizado por el verificador
            "errores_frecuentes": _top_errors(runs),
        },
        "usuarios": {
            "tiempo_medio_por_decision_s": mean(dec_times),
            "encuestas": len(surveys),
            "sus_medio": mean([s["sus_score"] for s in surveys]),
            "claridad_media": mean([s["clarity"] for s in surveys if s["clarity"]]),
            "fidelidad_media": mean([s["fidelity"] for s in surveys if s["fidelity"]]),
        },
        "aceptacion_por_eje": per_axis,
        "guardrails": guard,
    }


def _top_errors(runs: list[dict], k: int = 5) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for r in runs:
        for attempt in json.loads(r["errors"] or "[]"):
            for e in attempt:
                key = e.split(":")[0][:60]
                counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:k]

import json
import random

import pytest

from paisajia.agent import SimulatedLLM, build_template, explain, verify

LOG = [
    {"id": "R1-1", "ronda": 1, "eje": "hora", "variante": "atardecer", "decision": "aceptada", "motivo": "me encanta la luz"},
    {"id": "R1-2", "ronda": 1, "eje": "estilo", "variante": "óleo impresionista", "decision": "rechazada", "motivo": "muy artificial"},
    {"id": "R1-3", "ronda": 1, "eje": "estacion", "variante": "invierno", "decision": "rechazada", "motivo": None},
    {"id": "R1-4", "ronda": 1, "eje": "clima", "variante": "niebla", "decision": "aceptada", "motivo": None},
    {"id": "R1-5", "ronda": 1, "eje": "epoca", "variante": "años 70", "decision": "aceptada", "motivo": None},
]
GOOD = {"aceptadas": ["R1-1", "R1-4", "R1-5"], "rechazadas": ["R1-2", "R1-3"],
        "resumen": "El usuario aceptó atardecer (R1-1), niebla (R1-4) y años 70 (R1-5). "
                   "Rechazó óleo impresionista (R1-2) e invierno (R1-3), y escribió «muy artificial» sobre R1-2."}


def test_good_output_passes():
    res = verify(json.dumps(GOOD, ensure_ascii=False), LOG)
    assert res.valid, res.errors


def test_template_always_passes():
    rng = random.Random(0)
    from eval_agent import random_log

    for _ in range(200):
        log, comments = random_log(rng)
        res = verify(build_template(log, comments), log, comments)
        assert res.valid, res.errors


@pytest.mark.parametrize("mutate, fragment", [
    (lambda o: "{no es json", "JSON inválido"),
    (lambda o: {**o, "extra": 1}, "Claves"),
    (lambda o: {**o, "aceptadas": ["R1-1", "R1-4"]}, "Faltan aceptadas"),
    (lambda o: {**o, "rechazadas": o["rechazadas"] + ["R1-1"]}, "ambas listas"),
    (lambda o: {**o, "resumen": o["resumen"] + " Le gustó mucho invierno."}, "Contradicción"),
    (lambda o: {**o, "resumen": o["resumen"] + " Le gustó R1-2."}, "Contradicción"),
    (lambda o: {**o, "resumen": o["resumen"] + " Rechazó atardecer."}, "Contradicción"),
    (lambda o: {**o, "resumen": o["resumen"] + " También probó acuarela."}, "acuarela"),
    (lambda o: {**o, "resumen": o["resumen"] + " Mostró interés por la aurora boreal."}, "aurora boreal"),
    (lambda o: {**o, "resumen": o["resumen"] + " Rechazó R1-3 porque era demasiado oscura."}, "motivo"),
    (lambda o: {**o, "resumen": o["resumen"] + " Escribió «quiero más color»."}, "no escribió"),
    (lambda o: {**o, "resumen": o["resumen"] + " Ver R9-9."}, "R9-9"),
])
def test_errors_detected(mutate, fragment):
    out = mutate(dict(GOOD))
    res = verify(out if isinstance(out, str) else json.dumps(out, ensure_ascii=False), LOG)
    assert not res.valid
    assert any(fragment in e for e in res.errors), res.errors


def test_explain_with_retries_and_template():
    ok = explain(LOG, {}, SimulatedLLM(error_rate=0.0, seed=1))
    assert ok.first_try_valid and not ok.used_template
    always_bad = SimulatedLLM(error_rate=1.0, fix_rate=0.0, seed=2)
    fb = explain(LOG, {}, always_bad)
    assert fb.used_template and fb.attempts == 3
    assert verify(fb.output, LOG).valid


def test_summary_requires_all_decisions():
    log = [dict(LOG[0], decision=None)]
    with pytest.raises(ValueError):
        explain(log, {}, SimulatedLLM())

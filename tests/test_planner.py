from paisajia.config import AXES, AXIS_NAMES, CUSTOM_AXIS, N_VARIATIONS, base_variant
from paisajia.planner import ThompsonPlanner, initial_round


def test_full_package_uses_base_variants():
    slots = initial_round(seed=1)
    assert len(slots) == N_VARIATIONS
    assert [(s.axis, s.variant) for s in slots] == [(a, base_variant(a)) for a in AXIS_NAMES]


def test_individual_axes_always_five():
    slots = initial_round(["hora"], {"hora": "hora azul"}, seed=1)
    assert len(slots) == 5 and all(s.axis == "hora" for s in slots)
    assert slots[0].variant == "hora azul"
    assert len({s.variant for s in slots[:4]}) == 4  # recorre todas las variantes

    slots = initial_round(["clima", "epoca"], seed=2)
    assert len(slots) == 5 and {s.axis for s in slots} == {"clima", "epoca"}


def test_custom_instruction_slot():
    slots = initial_round(["hora"], custom_text="luz de luna", seed=3)
    assert len(slots) == 5 and slots[0].axis == CUSTOM_AXIS
    only = initial_round([], custom_text="luz de luna", seed=3)
    assert len(only) == 5 and all(s.axis == CUSTOM_AXIS for s in only)


def test_exploit_and_explore():
    planner = ThompsonPlanner(seed=0)
    prev = [
        {"slot": 1, "axis": "hora", "variant": "atardecer", "decision": "aceptada"},
        {"slot": 2, "axis": "estacion", "variant": "invierno", "decision": "rechazada"},
    ]
    stats = {("hora", "atardecer"): (1, 0), ("estacion", "invierno"): (0, 1)}
    slots, table = planner.next_round(prev, stats)
    assert slots[0].variant == "atardecer" and "aceptada" in slots[0].reason
    assert slots[1].axis == "estacion" and slots[1].variant != "invierno"
    assert slots[1].reason == f"{slots[1].variant}: alternativa a invierno, rechazada"
    assert table[1].accion == "explorar" and table[0].accion == "explotar"


def test_thompson_prefers_untested_over_rejected():
    planner = ThompsonPlanner(seed=5)
    stats = {("estacion", "otoño"): (0, 6)}
    picks = [planner._explore("estacion", "invierno", stats, set())[0].variant for _ in range(200)]
    assert picks.count("otoño") < 20  # casi nunca se elige una variante rechazada muchas veces
    assert set(picks) <= set(AXES["estacion"]["variants"]) - {"invierno"}

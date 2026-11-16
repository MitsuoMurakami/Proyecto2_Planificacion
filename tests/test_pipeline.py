import io
import json

import numpy as np
import pytest
from PIL import Image

from paisajia.agent import SimulatedLLM
from paisajia.metrics import summary, sus_score
from paisajia.preprocessing import preprocess
from paisajia.session import StudioSession


def _with_exif(img: Image.Image) -> Image.Image:
    exif = Image.Exif()
    exif[0x010F] = "CamaraX"  # Make
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    return Image.open(io.BytesIO(buf.getvalue()))


def test_preprocess_strips_exif_and_resizes(landscape):
    res = preprocess(_with_exif(landscape.resize((800, 600))))
    assert res.ok, res.reasons
    assert res.had_exif and res.image.size == (1024, 1024)
    assert not res.image.info.get("exif") and len(res.image_hash) == 64


def test_preprocess_rejects_blurry():
    flat = Image.fromarray(np.full((600, 600, 3), 128, np.uint8))
    res = preprocess(flat)
    assert not res.ok and any("borrosa" in r for r in res.reasons)


def test_full_session(landscape, registry):
    s = StudioSession(registry, "mock", SimulatedLLM(seed=0), seed=0)
    assert s.start(landscape).ok
    r1 = s.first_round()
    assert len(r1) == 5 and len({v.var_id for v in r1}) == 5
    assert all(Image.open(v.labeled_path).text["paisajia:provenance"] for v in r1)

    with pytest.raises(ValueError):
        s.summarize()  # aún hay pendientes
    s.decide("R1-1", "aceptada", "me encanta la luz")
    s.decide("R1-3", "rechazada", "muy artificial")
    regen = s.regenerate("R1-2")
    assert regen.version == 2
    assert s.decide_pending("rechazada", "ninguna me convenció del todo") == 3
    exp, table = s.summarize()
    assert exp.output["aceptadas"] == ["R1-1"]
    assert sorted(exp.output["rechazadas"]) == ["R1-2", "R1-3", "R1-4", "R1-5"]
    assert table[0].accion == "explotar" and table[2].accion == "explorar"

    r2 = s.next_round()
    assert len(r2) == 5 and r2[0].variant == r1[0].variant and r2[0].seed != r1[0].seed
    assert "alternativa a" in r2[2].plan_reason
    for v in r2:
        s.decide(v.var_id, "aceptada")
    exp2, _ = s.summarize()
    assert len(exp2.output["aceptadas"]) == 6

    data = json.loads((registry.session_dir(s.sid) / "ronda_2.json").read_text(encoding="utf-8"))
    assert len(data["decisions"]) == 5 and data["explicacion"]["aceptadas"]
    assert registry.query("SELECT image_hash FROM sessions")[0]["image_hash"] == s.pre.image_hash

    m = summary(registry)
    assert m["iteracion"]["aceptacion_ronda1"] == pytest.approx(0.2)
    assert m["iteracion"]["aceptacion_rondas_2+"] == pytest.approx(1.0)

    s.close()
    assert not list(registry.session_dir(s.sid).glob("*.png"))


def test_custom_instruction_blocked(landscape, registry):
    s = StudioSession(registry, "mock", SimulatedLLM(), seed=0)
    s.start(landscape)
    with pytest.raises(PermissionError):
        s.first_round(custom_text="añade turistas en la orilla")
    assert registry.query("SELECT COUNT(*) AS n FROM guardrail_events")[0]["n"] == 1


def test_sus():
    assert sus_score([5, 1] * 5) == 100
    assert sus_score([3] * 10) == 50

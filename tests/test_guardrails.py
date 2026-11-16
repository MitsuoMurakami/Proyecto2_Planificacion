import json

import pytest
from PIL import Image

from paisajia.guardrails import check_instruction, provenance_info, stamp_ai_label


@pytest.mark.parametrize("text", [
    "añade una persona caminando", "pon un letrero con mi nombre", "agrega una casa junto al lago",
    "convierte el bosque en un incendio forestal", "quita la etiqueta de IA", "add people on the beach",
    "un tsunami en la costa", "escena con sangre", "que parezca real sin marca",
])
def test_blocked(text):
    assert not check_instruction(text).allowed


@pytest.mark.parametrize("text", [
    "luz de luna llena reflejada en el agua", "colores pastel suaves al amanecer", "atmósfera de sueño",
    "tonos turquesa en el cielo", "llamas pastando en la puna",
])
def test_allowed(text):
    assert check_instruction(text).allowed, check_instruction(text).message


def test_empty_and_long():
    assert not check_instruction("   ").allowed
    assert not check_instruction("a" * 400).allowed


def test_label_and_provenance(tmp_path):
    img = Image.new("RGB", (512, 512), (10, 120, 60))
    out = stamp_ai_label(img)
    assert out.size == img.size and out.tobytes() != img.tobytes()
    p = tmp_path / "x.png"
    out.save(p, pnginfo=provenance_info({"seed": 3}))
    meta = json.loads(Image.open(p).text["paisajia:provenance"])
    assert meta["ai_generated"] and meta["seed"] == 3

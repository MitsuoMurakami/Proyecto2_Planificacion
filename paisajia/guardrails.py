"""Guardrails de instrucciones, etiquetado visible y metadatos de procedencia.

- Instrucciones del usuario: se bloquean pedidos de añadir personas, texto,
  construcciones, contenido sexual o violento, desastres sobre lugares reales
  o la eliminación de la etiqueta de IA.
- Transparencia: toda variación lleva la etiqueta visible «Generado con IA» y
  metadatos de procedencia (chunks de texto PNG).
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone

from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

from .config import AI_LABEL


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


# Cada categoría: lista de patrones (sobre texto en minúsculas y sin tildes).
BLOCKED: dict[str, list[str]] = {
    "personas": [
        r"\bpersonas?\b", r"\bgente\b", r"\bhombres?\b", r"\bmujer(es)?\b", r"\bnin[oa]s?\b",
        r"\bturistas?\b", r"\bcaminantes?\b", r"\bsilueta", r"\brostros?\b", r"\bcaras?\b",
        r"\bpeople\b", r"\bperson\b", r"\bman\b", r"\bwoman\b", r"\bchild", r"\bhuman", r"\bface",
        r"\bhikers?\b", r"\bselfie",
    ],
    "texto": [
        r"\btexto\b", r"\bletras?\b", r"\bletreros?\b", r"\bcarteles?\b", r"\bpalabras?\b", r"\bfirma\b",
        r"\blogos?\b", r"\btext\b", r"\bletters?\b", r"\bsigns?\b", r"\bwords?\b", r"\bcaption",
        r"\bescribe\b", r"\bwrite\b",
    ],
    "construcciones": [
        r"\bcasas?\b", r"\bedificios?\b", r"\bpuentes?\b", r"\btorres?\b", r"\bcastillos?\b",
        r"\bcarreteras?\b", r"\bciudad\b", r"\bbuildings?\b", r"\bhouses?\b", r"\bbridges?\b",
        r"\btowers?\b", r"\bcastles?\b", r"\broads?\b", r"\bcity\b", r"\bconstrucci",
    ],
    "contenido sexual o violento": [
        r"\bdesnud", r"\bsexual", r"\berotic", r"\bnude", r"\bnsfw", r"\bsangre\b", r"\bblood",
        r"\bcadaver", r"\bcorpse", r"\barmas?\b", r"\bweapons?\b", r"\bguns?\b", r"\bviolen",
        r"\bmuert", r"\bdead\b", r"\bgore\b",
    ],
    "desastres": [
        r"\bincendio", r"\bfuego\b", r"\ben llamas\b",r"\bterremoto", r"\btsunami", r"\binundaci",
        r"\bexplosi", r"\bbomba", r"\bdestrui", r"\bdestrucci", r"\bhuracan", r"\berupci",
        r"\bwildfire", r"\bfire\b", r"\bburning", r"\bflood", r"\bearthquake", r"\bexplosion",
        r"\bdestroy", r"\bhurricane", r"\btornado", r"\bdisaster", r"\bdesastre", r"\bapocalip",
        r"\bguerra\b", r"\bwar\b",
    ],
    "eliminar etiqueta de IA": [
        r"(quita|quitar|elimina|eliminar|borra|borrar|remove|delete|hide|oculta|ocultar|sin)\b.{0,25}"
        r"(etiqueta|label|marca|watermark|generado con ia|ai label)",
        r"\bno (pongas|incluyas|muestres) .{0,20}(etiqueta|marca)",
        r"\bque parezca real\b", r"\bfoto real\b", r"\bauthentic\b", r"\bnot ai\b",
    ],
}
_COMPILED = {cat: [re.compile(p) for p in pats] for cat, pats in BLOCKED.items()}

MAX_PROMPT_CHARS = 300


@dataclass
class GuardrailResult:
    allowed: bool
    category: str | None = None
    match: str | None = None
    message: str = ""


def check_instruction(text: str) -> GuardrailResult:
    if not text or not text.strip():
        return GuardrailResult(False, "vacía", None, "La instrucción está vacía.")
    if len(text) > MAX_PROMPT_CHARS:
        return GuardrailResult(False, "longitud", None, f"La instrucción supera {MAX_PROMPT_CHARS} caracteres.")
    norm = _norm(text)
    for cat, pats in _COMPILED.items():
        for p in pats:
            m = p.search(norm)
            if m:
                return GuardrailResult(
                    False, cat, m.group(0),
                    f"Instrucción bloqueada ({cat}: «{m.group(0)}»). PaisajIA solo modifica luz, estilo, "
                    "estación, clima o época; no añade personas, texto, construcciones ni desastres.",
                )
    return GuardrailResult(True, message="Instrucción permitida.")


# ---------------------------------------------------------------------------
# Transparencia
# ---------------------------------------------------------------------------
def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def stamp_ai_label(img: Image.Image, text: str = AI_LABEL) -> Image.Image:
    """Añade la etiqueta visible «Generado con IA» en la esquina inferior derecha."""
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out, "RGBA")
    size = max(14, out.width // 40)
    font = _font(size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = size // 2
    x1, y1 = out.width - tw - 3 * pad, out.height - th - 3 * pad
    draw.rounded_rectangle([x1, y1, out.width - pad, out.height - pad], radius=pad, fill=(0, 0, 0, 160))
    draw.text((x1 + pad, y1 + pad // 2), text, font=font, fill=(255, 255, 255, 235))
    return out


def provenance_info(meta: dict) -> PngInfo:
    """Metadatos de procedencia en el PNG (inspirado en IPTC DigitalSourceType / C2PA)."""
    info = PngInfo()
    record = {
        "generator": "PaisajIA",
        "digital_source_type": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
        "ai_generated": True,
        "label": AI_LABEL,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **meta,
    }
    info.add_text("paisajia:provenance", json.dumps(record, ensure_ascii=False))
    info.add_text("Software", "PaisajIA")
    info.add_text("Comment", f"{AI_LABEL}. Imagen sintética derivada de una foto del usuario.")
    return info


def save_with_provenance(img: Image.Image, path, meta: dict) -> None:
    img.save(path, format="PNG", pnginfo=provenance_info(meta))

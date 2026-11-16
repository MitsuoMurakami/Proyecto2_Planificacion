"""Verificador determinista del resumen del agente.

Comprueba tres cosas:
1. que el JSON sea válido (claves y tipos exactos);
2. que las listas `aceptadas` y `rechazadas` coincidan exactamente con el registro;
3. que el `resumen` no mencione ejes/variantes inexistentes en la sesión, no
   contradiga decisiones y no cite motivos que el usuario no escribió.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

from ..config import AXES, CUSTOM_AXIS

EXPECTED_KEYS = {"aceptadas", "rechazadas", "resumen"}
ID_RE = re.compile(r"\bR\d+-\d+\b")
QUOTE_RE = re.compile(r"[«\"“][^»\"”]*[»\"”]")

# Transformaciones plausibles que NO existen en PaisajIA: si aparecen, el LLM inventó un eje.
INVENTED_TERMS = [
    "fauna", "animales", "aurora boreal", "arcoiris", "arco iris", "volcan", "cyberpunk", "anime", "pixel art",
    "acrilico", "carboncillo", "surrealista", "vaporwave", "fantasia", "futurista", "años 80", "años 60",
    "años 30", "años 40", "años 2000", "eclipse", "nublado parcial", "granizo", "arena", "neon",
    "iluminacion artificial", "temporada de lluvias",
]
# Palabras de motivo: solo pueden aparecer si el usuario escribió algo con esa raíz.
REASON_STEMS = [
    "oscur", "borros", "artificial", "irreal", "saturad", "apagad", "feo", "fea", "bonit", "hermos",
    "realist", "contrast", "ruido", "granulad", "falso", "falsa", "extrañ", "exagerad", "sobreexpuest",
    "deform", "distorsion", "aburrid", "llamativ", "elegant", "nitid", "pixel", "caricatur",
]
ACCEPT_PAT = re.compile(r"acept|gust|prefi|prefer|eligi|elegid|conserv|aprob|quedo con|se quedo|valor")
REJECT_PAT = re.compile(
    r"rechaz|descart|no (le |te |les )?(gusto|gustaron|convenc)|desestim|evit|no fue(ron)? acept|no acept|no eligi"
)


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _contains(text_n: str, term: str) -> bool:
    t = re.escape(norm(term))
    return re.search(rf"(?<![a-z0-9]){t}(?![a-z0-9])", text_n) is not None


@dataclass
class VerificationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    parsed: dict | None = None

    def feedback(self) -> str:
        return "\n".join(f"- {e}" for e in self.errors)


def expected_lists(log: list[dict]) -> tuple[list[str], list[str]]:
    acc = [r["id"] for r in log if r.get("decision") == "aceptada"]
    rej = [r["id"] for r in log if r.get("decision") == "rechazada"]
    return acc, rej


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.;!?])\s+|\n+", text) if s.strip()]


def verify(raw: str | dict, log: list[dict], package_comments: dict[int, str] | None = None) -> VerificationResult:
    errors: list[str] = []
    # 1. JSON válido -------------------------------------------------------
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            return VerificationResult(False, [f"JSON inválido: {exc}"])
    if not isinstance(data, dict):
        return VerificationResult(False, ["La salida debe ser un objeto JSON."])
    keys = set(data)
    if keys != EXPECTED_KEYS:
        errors.append(f"Claves esperadas {sorted(EXPECTED_KEYS)}, recibidas {sorted(keys)}.")
    acc, rej, resumen = data.get("aceptadas"), data.get("rechazadas"), data.get("resumen")
    if not isinstance(acc, list) or not all(isinstance(x, str) for x in acc):
        errors.append("'aceptadas' debe ser una lista de IDs (texto).")
        acc = []
    if not isinstance(rej, list) or not all(isinstance(x, str) for x in rej):
        errors.append("'rechazadas' debe ser una lista de IDs (texto).")
        rej = []
    if not isinstance(resumen, str) or not resumen.strip():
        errors.append("'resumen' debe ser un texto no vacío.")
        resumen = ""
    if errors:
        return VerificationResult(False, errors, data)

    # 2. Listas exactas ---------------------------------------------------
    exp_acc, exp_rej = expected_lists(log)
    if len(acc) != len(set(acc)) or len(rej) != len(set(rej)):
        errors.append("Hay IDs duplicados en las listas.")
    if set(acc) & set(rej):
        errors.append(f"IDs en ambas listas: {sorted(set(acc) & set(rej))}.")
    missing_a, extra_a = set(exp_acc) - set(acc), set(acc) - set(exp_acc)
    missing_r, extra_r = set(exp_rej) - set(rej), set(rej) - set(exp_rej)
    if missing_a:
        errors.append(f"Faltan aceptadas: {sorted(missing_a)}.")
    if extra_a:
        errors.append(f"Marcadas como aceptadas sin estarlo en el registro: {sorted(extra_a)}.")
    if missing_r:
        errors.append(f"Faltan rechazadas: {sorted(missing_r)}.")
    if extra_r:
        errors.append(f"Marcadas como rechazadas sin estarlo en el registro: {sorted(extra_r)}.")

    # 3. Resumen fiel ------------------------------------------------------
    errors += check_summary(resumen, log, package_comments or {})
    return VerificationResult(not errors, errors, data)


def check_summary(resumen: str, log: list[dict], package_comments: dict[int, str]) -> list[str]:
    errors: list[str] = []
    rn = norm(resumen)
    ids = {r["id"]: r for r in log}
    session_axes = {r["eje"] for r in log}
    session_variants = {r["variante"] for r in log if r["eje"] != CUSTOM_AXIS}
    motives = [r["motivo"] for r in log if r.get("motivo")] + list(package_comments.values())
    motives_n = norm(" ".join(motives))
    # Vocabulario permitido además de los motivos: el texto de las instrucciones propias.
    allowed_n = norm(" ".join([motives_n] + [r.get("instruccion") or "" for r in log]))

    # 3a. Ejes o variantes inexistentes en la sesión.
    for axis, spec in AXES.items():
        if axis not in session_axes and _contains(rn, spec["label"]):
            errors.append(f"El resumen menciona el eje «{spec['label']}», que no aparece en la sesión.")
        for v in spec["variants"]:
            if v not in session_variants and _contains(rn, v) and not _contains(allowed_n, v):
                errors.append(f"El resumen menciona la variante «{v}», que no se generó en la sesión.")
    for term in INVENTED_TERMS:
        if _contains(rn, term) and not _contains(allowed_n, term):
            errors.append(f"El resumen menciona «{term}», que no es un eje ni una variante de PaisajIA.")
    for mid in set(ID_RE.findall(resumen)):
        if mid not in ids:
            errors.append(f"El resumen cita el ID {mid}, que no existe en el registro.")

    # 3b. Contradicciones con las decisiones (análisis por oración).
    counts: dict[str, list[int]] = {}
    for r in log:
        c = counts.setdefault(r["variante"], [0, 0])
        if r.get("decision") == "aceptada":
            c[0] += 1
        elif r.get("decision") == "rechazada":
            c[1] += 1
    for sent in _sentences(resumen):
        sent = QUOTE_RE.sub(" ", sent)  # lo citado es texto del usuario, no una afirmación del agente
        sn = norm(sent)
        rej_pol = bool(REJECT_PAT.search(sn))
        acc_pol = bool(ACCEPT_PAT.search(REJECT_PAT.sub(" ", sn)))
        if rej_pol == acc_pol:  # neutra o mixta: no se puede atribuir polaridad
            continue
        for mid in set(ID_RE.findall(sent)):
            dec = ids.get(mid, {}).get("decision")
            if dec and ((acc_pol and dec == "rechazada") or (rej_pol and dec == "aceptada")):
                errors.append(f"Contradicción: «{sent.strip()}» no coincide con la decisión de {mid} ({dec}).")
        for variant, (n_acc, n_rej) in counts.items():
            if variant == "instrucción propia" or not _contains(sn, variant):
                continue
            if acc_pol and n_acc == 0 and n_rej > 0:
                errors.append(f"Contradicción: el resumen dice que «{variant}» gustó, pero solo fue rechazada.")
            if rej_pol and n_rej == 0 and n_acc > 0:
                errors.append(f"Contradicción: el resumen dice que «{variant}» fue rechazada, pero solo fue aceptada.")

    # 3c. Motivos no escritos por el usuario.
    for quote in re.findall(r"[«\"“]([^»\"”]{3,})[»\"”]", resumen):
        if norm(quote).strip(" .,") not in motives_n:
            errors.append(f"El resumen cita «{quote}», que el usuario no escribió.")
    unquoted = QUOTE_RE.sub(" ", rn)
    for stem in REASON_STEMS:
        if re.search(rf"\b{stem}", unquoted) and not re.search(rf"\b{stem}", motives_n):
            errors.append(f"El resumen atribuye un motivo («{stem}…») que el usuario no escribió.")
    return list(dict.fromkeys(errors))

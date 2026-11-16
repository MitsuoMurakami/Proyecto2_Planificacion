"""Clientes LLM del agente de explicación.

- `ClaudeLLM`: Claude (API de Anthropic) con salida estructurada (JSON schema).
- `OllamaLLM`: LLM local vía Ollama (por defecto Qwen 2.5 3B), gratis y sin enviar datos fuera.
- `SimulatedLLM`: LLM simulado con tasa de error configurable. Sirve para el
  modo sin conexión y para evaluar el verificador en 100 sesiones simuladas sin
  coste; sus errores imitan los fallos típicos de un LLM (omitir IDs, invertir
  decisiones, inventar ejes o motivos, JSON roto).

El agente recibe solo el registro (IDs, ejes, variantes, decisiones, motivos),
nunca la foto.
"""
from __future__ import annotations

import json
import os
import random
from abc import ABC, abstractmethod

from ..config import SETTINGS

SYSTEM_PROMPT = """Eres el agente de explicación de PaisajIA, un estudio de paisaje con feedback humano.
Recibes SOLO el registro de decisiones de una sesión (nunca las imágenes): cada variación tiene un ID
(p. ej. R2-3 = ronda 2, posición 3), un eje (hora, estilo, estacion, clima, epoca o personalizada), una
variante, la decisión del usuario (aceptada o rechazada) y, a veces, un motivo escrito por el usuario.

Devuelve un JSON con tres campos:
- "aceptadas": lista exacta de los IDs aceptados;
- "rechazadas": lista exacta de los IDs rechazados;
- "resumen": 3 a 6 oraciones en español que describan qué se aceptó y qué se rechazó y el patrón de
  preferencias a lo largo de las rondas (qué ejes y variantes se repiten entre las aceptadas y cuáles
  se descartan).

Reglas estrictas (un verificador automático las comprueba):
1. No inventes ejes, variantes ni IDs que no estén en el registro.
2. No contradigas las decisiones: nunca digas que gustó algo rechazado ni al revés.
3. No atribuyas motivos ni describas cómo se ven las imágenes. Si mencionas un motivo, cítalo
   textualmente entre comillas angulares «...» tal como lo escribió el usuario.
4. Cuando una oración mezcle aceptadas y rechazadas, sepáralas en oraciones distintas.
5. Escribe los nombres de las variantes tal como aparecen en el registro."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "aceptadas": {"type": "array", "items": {"type": "string"}},
        "rechazadas": {"type": "array", "items": {"type": "string"}},
        "resumen": {"type": "string"},
    },
    "required": ["aceptadas", "rechazadas", "resumen"],
    "additionalProperties": False,
}


class LLMError(RuntimeError):
    pass


def render_log(log: list[dict], package_comments: dict[int, str]) -> str:
    payload = {
        "registro": [{k: v for k, v in r.items() if v is not None and k != "via"} for r in log],
        "comentarios_de_paquete": {f"ronda {k}": v for k, v in package_comments.items()},
    }
    return "Registro de la sesión:\n" + json.dumps(payload, ensure_ascii=False, indent=1)


class BaseLLM(ABC):
    name = "base"

    @abstractmethod
    def summarize(self, log: list[dict], package_comments: dict[int, str], feedback: list[str]) -> str:
        """Devuelve el texto crudo (JSON esperado). `feedback`: errores del verificador en intentos previos."""

    def reset(self) -> None:
        """Comienza una conversación nueva (una por resumen)."""


class ClaudeLLM(BaseLLM):
    name = "claude"

    def __init__(self, model: str | None = None, effort: str = "low"):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model or SETTINGS.llm_model
        self.effort = effort
        self.messages: list = []

    def reset(self) -> None:
        self.messages = []

    def _create(self, use_fallbacks: bool):
        kwargs = dict(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=self.messages,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        )
        if use_fallbacks:
            # Si el clasificador de seguridad rechaza la petición, la API la reintenta en otro modelo.
            return self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        return self.client.messages.create(**kwargs)

    def summarize(self, log, package_comments, feedback):
        if not self.messages:
            self.messages = [{"role": "user", "content": render_log(log, package_comments)}]
        elif feedback:
            self.messages.append({
                "role": "user",
                "content": "El verificador rechazó tu respuesta por estos motivos:\n"
                           + "\n".join(f"- {e}" for e in feedback)
                           + "\nCorrige el JSON completo respetando el registro.",
            })
        a = self.anthropic
        try:
            try:
                resp = self._create(use_fallbacks=True)
            except (TypeError, a.BadRequestError):
                resp = self._create(use_fallbacks=False)  # SDK o plataforma sin `fallbacks`
        except a.AuthenticationError as exc:
            raise LLMError(f"Credenciales de Anthropic inválidas: {exc}") from exc
        except a.RateLimitError as exc:
            raise LLMError(f"Límite de uso alcanzado: {exc}") from exc
        except a.APIStatusError as exc:
            raise LLMError(f"Error de la API ({exc.status_code}): {exc.message}") from exc
        except a.APIConnectionError as exc:
            raise LLMError(f"Sin conexión con la API: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise LLMError("El modelo declinó la petición.")
        # Se conserva el contenido completo (incluidos bloques de razonamiento) para el reintento.
        self.messages.append({"role": "assistant", "content": resp.content})
        text = next((b.text for b in resp.content if b.type == "text"), "")
        if not text:
            raise LLMError(f"Respuesta vacía (stop_reason={resp.stop_reason}).")
        return text


class OllamaLLM(BaseLLM):
    """LLM local servido por Ollama (por defecto Qwen 2.5 3B, `qwen2.5:3b`).

    Gratis y privado: el registro no sale de la máquina. Se usa la API HTTP de
    Ollama con `format` = JSON schema (salida estructurada) y temperatura 0.
    El modelo cuantizado ocupa ~2 GB de VRAM (o corre en CPU si no hay GPU).
    """

    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 120.0):
        self.model = model or SETTINGS.ollama_model
        self.host = (host or SETTINGS.ollama_host).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.timeout = timeout
        self.name = f"ollama:{self.model}"
        self.messages: list[dict] = []

    def reset(self) -> None:
        self.messages = []

    def _post(self, path: str, payload: dict) -> dict:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(self.host + path, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            if exc.code == 404:
                raise LLMError(f"Modelo '{self.model}' no descargado. Ejecuta: ollama pull {self.model}") from exc
            raise LLMError(f"Error de Ollama ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise LLMError(f"Ollama no responde en {self.host} ({exc}). ¿Está abierto `ollama serve`?") from exc

    @staticmethod
    def _options() -> dict:
        opts = {"temperature": 0, "num_ctx": 8192}
        if SETTINGS.ollama_num_gpu.strip():
            opts["num_gpu"] = int(SETTINGS.ollama_num_gpu)  # 0 = Qwen solo en CPU
        return opts

    def summarize(self, log, package_comments, feedback):
        if not self.messages:
            self.messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": render_log(log, package_comments)},
            ]
        elif feedback:
            self.messages.append({
                "role": "user",
                "content": "El verificador rechazó tu respuesta por estos motivos:\n"
                           + "\n".join(f"- {e}" for e in feedback)
                           + "\nCorrige el JSON completo respetando el registro.",
            })
        resp = self._post("/api/chat", {
            "model": self.model,
            "messages": self.messages,
            "stream": False,
            "format": OUTPUT_SCHEMA,
            "options": self._options(),
        })
        text = (resp.get("message") or {}).get("content", "")
        if not text:
            raise LLMError(f"Respuesta vacía de Ollama: {resp.get('error') or resp.get('done_reason')}")
        self.messages.append({"role": "assistant", "content": text})
        return text


class SimulatedLLM(BaseLLM):
    """LLM simulado con errores inyectados (para modo sin conexión y evaluación)."""

    name = "simulado"
    ERROR_TYPES = ["omitir_id", "invertir_decision", "eje_inventado", "motivo_inventado", "contradiccion", "json_roto"]

    def __init__(self, error_rate: float = 0.0, fix_rate: float = 0.8, seed: int | None = None):
        self.error_rate, self.fix_rate = error_rate, fix_rate
        self.rng = random.Random(seed)
        self.last_errors: list[str] = []

    def reset(self) -> None:
        self.last_errors = []

    def summarize(self, log, package_comments, feedback):
        from .template import axis_label

        acc = [r for r in log if r.get("decision") == "aceptada"]
        rej = [r for r in log if r.get("decision") == "rechazada"]
        rounds = sorted({r["ronda"] for r in log})
        parts = []
        if acc:
            parts.append("El usuario aceptó " + ", ".join(f"{r['variante']} ({r['id']})" for r in acc) + ".")
        if rej:
            parts.append("Rechazó " + ", ".join(f"{r['variante']} ({r['id']})" for r in rej) + ".")
        motives = [r for r in log if r.get("motivo")]
        if motives:
            parts.append("Sus motivos fueron: " + "; ".join(f"«{r['motivo']}»" for r in motives) + ".")
        axes_acc = sorted({axis_label(r["eje"]) for r in acc})
        if axes_acc and len(rounds) > 1:
            parts.append(f"A lo largo de {len(rounds)} rondas, las variaciones conservadas se concentran en: {', '.join(axes_acc)}.")
        out = {"aceptadas": [r["id"] for r in acc], "rechazadas": [r["id"] for r in rej], "resumen": " ".join(parts)}

        # Inyección de errores: en el primer intento con prob. error_rate; tras feedback, se corrige con prob. fix_rate.
        if feedback and self.rng.random() < self.fix_rate:
            self.last_errors = []
        elif not feedback and self.rng.random() < self.error_rate:
            self.last_errors = [self.rng.choice(self.ERROR_TYPES)]
        for err in self.last_errors:
            out = self._inject(err, out, acc, rej)
            if isinstance(out, str):
                return out
        return json.dumps(out, ensure_ascii=False)

    def _inject(self, err, out, acc, rej):
        if err == "json_roto":
            return json.dumps(out, ensure_ascii=False)[:-7]
        if err == "omitir_id" and (out["aceptadas"] or out["rechazadas"]):
            key = "aceptadas" if out["aceptadas"] else "rechazadas"
            out[key] = out[key][1:]
        elif err == "invertir_decision" and acc:
            out["aceptadas"].remove(acc[0]["id"])
            out["rechazadas"].append(acc[0]["id"])
        elif err == "eje_inventado":
            out["resumen"] += " También mostró interés por la variante de aurora boreal."
        elif err == "motivo_inventado":
            out["resumen"] += " Rechazó las imágenes porque le parecían demasiado oscuras."
        elif err == "contradiccion" and rej:
            out["resumen"] += f" Le gustó especialmente {rej[0]['id']}."
        return out


def get_llm(kind: str | None = None, **kwargs) -> BaseLLM:
    kind = (kind or SETTINGS.llm).lower()
    if kind == "auto":
        has_creds = any(os.getenv(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"))
        kind = "claude" if has_creds else "simulado"
    if kind == "claude":
        return ClaudeLLM(**kwargs)
    if kind in ("ollama", "qwen"):
        return OllamaLLM(**kwargs)
    return SimulatedLLM(**kwargs)

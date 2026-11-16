"""Orquestador de una sesión de PaisajIA (flujo completo de la Figura 1).

foto -> preprocesamiento -> [generación + QC -> interacción -> registro ->
agente + verificador -> planificación] -> nueva ronda ...
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

from PIL import Image

from . import vision
from .agent import Explanation, explain, get_llm
from .agent.llm import BaseLLM
from .config import AXES, CUSTOM_AXIS, N_VARIATIONS, SETTINGS, prompt_for
from .generation import Generator, get_generator
from .guardrails import check_instruction, save_with_provenance, stamp_ai_label
from .planner import PlanRow, Slot, ThompsonPlanner, initial_round
from .preprocessing import PreprocessResult, preprocess
from .quality import generate_with_qc
from .registry import Registry

log = logging.getLogger(__name__)


@dataclass
class Variation:
    var_id: str
    round: int
    slot: int
    axis: str
    variant: str
    seed: int
    plan_reason: str
    labeled_path: str
    image: Image.Image            # sin etiqueta, para métricas
    dino: float | None
    dino_backend: str
    attempts: int
    warning: str | None = None
    custom_text: str | None = None
    version: int = 1
    decision: str | None = None
    reason: str | None = None
    shown_at: float = field(default_factory=time.time)

    @property
    def axis_label(self) -> str:
        return AXES[self.axis]["label"] if self.axis in AXES else "Instrucción propia"

    def caption(self) -> str:
        d = {"aceptada": "✅ aceptada", "rechazada": "❌ rechazada"}.get(self.decision or "", "⏳ pendiente")
        sim = f"sim. {self.dino:.2f}" if self.dino is not None else ""
        return f"{self.var_id} · {self.axis_label}: {self.variant} · {sim} · {d}"


class StudioSession:
    def __init__(self, registry: Registry | None = None, generator: Generator | str | None = None,
                 llm: BaseLLM | None = None, seed: int | None = None, simulated: bool = False):
        self.registry = registry or Registry()
        self.generator = generator if isinstance(generator, Generator) else get_generator(generator or SETTINGS.generator)
        self.llm = llm or get_llm()
        self.rng = random.Random(seed)
        self.planner = ThompsonPlanner(seed)
        self.simulated = simulated
        self.sid: str | None = None
        self.original: Image.Image | None = None
        self.pre: PreprocessResult | None = None
        self.round = 0
        self.variations: dict[str, Variation] = {}
        self.plan: list[Slot] | None = None
        self.plan_table: list[PlanRow] = []
        self.explanations: dict[int, Explanation] = {}
        self.round_seconds: dict[int, float] = {}
        self.lpips: dict[int, float] = {}

    # -- 1. preprocesamiento ----------------------------------------------
    def start(self, image: Image.Image, check_faces: bool = True) -> PreprocessResult:
        pre = preprocess(image, check_faces=check_faces)
        self.pre = pre
        if not pre.ok:
            self.registry.log_guardrail(None, "entrada", "imagen", "; ".join(pre.reasons))
            return pre
        self.original = pre.image
        self.sid = self.registry.create_session(pre.image_hash, pre.scene, self.generator.name, self.llm.name, self.simulated)
        return pre

    # -- 2-3. generación + control de calidad ------------------------------
    def first_round(self, axes: list[str] | None = None, variants: dict[str, str] | None = None,
                    custom_text: str | None = None) -> list[Variation]:
        if custom_text:
            g = check_instruction(custom_text)
            if not g.allowed:
                self.registry.log_guardrail(self.sid, "instruccion", g.category, custom_text)
                raise PermissionError(g.message)
        slots = initial_round(axes, variants, custom_text, seed=self.rng.randint(0, 2**31))
        return self._run_round(slots)

    def next_round(self) -> list[Variation]:
        if self.plan is None:
            raise RuntimeError("Primero hay que cerrar la ronda con el resumen.")
        slots, self.plan = self.plan, None
        return self._run_round(slots)

    def _run_round(self, slots: list[Slot]) -> list[Variation]:
        assert self.original is not None and self.sid is not None
        self.round += 1
        rnd = self.round
        self.registry.start_round(self.sid, rnd)
        t0 = time.perf_counter()
        out: list[Variation] = []
        for i, slot in enumerate(slots, start=1):
            out.append(self._make_variation(rnd, i, slot))
        elapsed = time.perf_counter() - t0
        lp, backend = vision.mean_pairwise_lpips([v.image for v in out])
        self.round_seconds[rnd], self.lpips[rnd] = elapsed, lp
        self.registry.finish_generation(self.sid, rnd, elapsed, lp, backend)
        assert len(out) == N_VARIATIONS
        return out

    def _make_variation(self, rnd: int, slot_i: int, slot: Slot, version: int = 1) -> Variation:
        var_id = f"R{rnd}-{slot_i}"
        candidates = [slot]
        if slot.axis in AXES:  # si la salida se descarta, se prueba otra variante del mismo eje
            candidates += [Slot(slot.axis, v, self.rng.randint(0, 2**31), f"{v}: sustituye a {slot.variant}, descartada por el guardrail")
                           for v in AXES[slot.axis]["variants"] if v != slot.variant]
        qc = None
        for cand in candidates:
            qc = generate_with_qc(self.generator, self.original, cand.axis, cand.variant, cand.seed, cand.custom_text)
            if not qc.discarded:
                slot = cand
                break
            self.registry.log_guardrail(self.sid, "salida", "descartada", f"{var_id} {cand.axis}/{cand.variant}: {qc.warning}")
        if qc is None or qc.discarded:
            raise RuntimeError(f"No se pudo generar una variación apta para {var_id}.")
        meta = {"session_hash": self.pre.image_hash[:16], "var_id": var_id, "axis": slot.axis, "variant": slot.variant,
                "seed": qc.seed, **self.generator.params()}
        path = self.registry.session_dir(self.sid) / f"{var_id}_v{version}.png"
        save_with_provenance(stamp_ai_label(qc.image), path, meta)
        v = Variation(var_id, rnd, slot_i, slot.axis, slot.variant, qc.seed, slot.reason, str(path), qc.image,
                      qc.dino, qc.dino_backend, qc.attempts, qc.warning, slot.custom_text, version)
        self.variations[var_id] = v
        self.registry.add_variation(self.sid, {
            "var_id": var_id, "round": rnd, "slot": slot_i, "version": version, "axis": slot.axis,
            "variant": slot.variant, "prompt": prompt_for(slot.axis, slot.variant, slot.custom_text), "seed": qc.seed,
            "attempts": qc.attempts, "dino": qc.dino, "dino_backend": qc.dino_backend,
            "clip_adherence": qc.clip_adherence, "warning": qc.warning, "plan_reason": slot.reason,
            "params": self.generator.params(), "gen_seconds": qc.seconds,
        })
        return v

    def regenerate(self, var_id: str) -> Variation:
        """Regenera una imagen puntual con otra semilla (misma variante)."""
        old = self.variations[var_id]
        self.registry.clear_decision(self.sid, var_id)
        slot = Slot(old.axis, old.variant, self.rng.randint(0, 2**31), f"{old.variant}: regenerada a pedido del usuario",
                    old.custom_text)
        return self._make_variation(old.round, old.slot, slot, version=old.version + 1)

    # -- 4-5. interacción y registro ---------------------------------------
    def current(self) -> list[Variation]:
        return sorted((v for v in self.variations.values() if v.round == self.round), key=lambda v: v.slot)

    def decide(self, var_id: str, decision: str, reason: str | None = None, via: str = "individual") -> None:
        v = self.variations[var_id]
        if v.round != self.round:
            raise ValueError("Solo se pueden decidir imágenes de la ronda actual.")
        v.decision, v.reason = decision, (reason or "").strip() or None
        self.registry.record_decision(self.sid, var_id, v.round, decision, v.reason, via, v.shown_at)

    def decide_pending(self, decision: str, comment: str | None = None) -> int:
        n = 0
        for v in self.current():
            if v.decision is None:
                self.decide(v.var_id, decision, None, via="paquete")
                n += 1
        if comment and comment.strip():
            self.registry.set_package_comment(self.sid, self.round, comment)
        return n

    def all_decided(self) -> bool:
        cur = self.current()
        return bool(cur) and all(v.decision for v in cur)

    # -- 6-7. explicación y planificación ----------------------------------
    def summarize(self) -> tuple[Explanation, list[PlanRow]]:
        if not self.all_decided():
            raise ValueError("El resumen se habilita solo cuando todas las imágenes tienen decisión.")
        log_rows = self.registry.decision_log(self.sid)
        comments = self.registry.package_comments(self.sid)
        exp = explain(log_rows, comments, self.llm)
        self.explanations[self.round] = exp
        self.registry.log_agent_run(self.sid, self.round, exp.llm_name, exp.attempts, exp.first_try_valid,
                                    exp.used_template, exp.errors, exp.output, exp.seconds)
        previous = [{"slot": v.slot, "axis": v.axis, "variant": v.variant, "decision": v.decision,
                     "custom_text": v.custom_text} for v in self.current()]
        self.plan, self.plan_table = self.planner.next_round(previous, self.registry.variant_stats(self.sid))
        self.registry.export_round_json(self.sid, self.round, {
            "explicacion": exp.output, "fuente_explicacion": exp.source,
            "plan_siguiente_ronda": [r.__dict__ for r in self.plan_table],
        })
        return exp, self.plan_table

    def close(self, delete_files: bool = True) -> None:
        if self.sid:
            self.registry.close_session(self.sid, delete_files=delete_files)


__all__ = ["StudioSession", "Variation", "CUSTOM_AXIS"]

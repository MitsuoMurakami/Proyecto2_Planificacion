"""Prueba rápida en GPU: genera una ronda con el generador elegido y un resumen con el LLM.

Mide tiempo por imagen, pico de VRAM y si el agente pasó el verificador. Sirve para
validar el entorno de Khipu (modelos descargados, CUDA, Ollama) antes de lanzar la app
o la evaluación completa.

    python scripts/smoke_test.py --generator flux --llm ollama
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from paisajia.agent import explain, get_llm  # noqa: E402
from paisajia.config import AXIS_NAMES, SETTINGS, base_variant  # noqa: E402
from paisajia.generation import get_generator  # noqa: E402
from paisajia.quality import generate_with_qc  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="flux", choices=["flux", "sd", "mock"])
    ap.add_argument("--llm", default="ollama", choices=["ollama", "claude", "simulado"])
    ap.add_argument("--image", type=Path, default=Path("examples/paisaje_demo.jpg"))
    ap.add_argument("--out", type=Path, default=Path("results/smoke"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {"generator": args.generator, "gen_size": SETTINGS.gen_size}

    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            report["gpu"] = f"{p.name} ({p.total_memory / 1024 ** 3:.1f} GB, capacidad {p.major}.{p.minor})"
            free, _ = torch.cuda.mem_get_info()
            report["vram_libre_al_inicio_gb"] = round(free / 1024 ** 3, 2)
        else:
            report["gpu"] = "sin CUDA"
    except ImportError:
        report["gpu"] = "torch no instalado"
    print(json.dumps(report, ensure_ascii=False), flush=True)

    t0 = time.perf_counter()
    gen = get_generator(args.generator, fallback_to_mock=False)
    report["carga_modelo_s"] = round(time.perf_counter() - t0, 1)
    report["params"] = gen.params()

    orig = Image.open(args.image).convert("RGB").resize((1024, 1024))
    t0 = time.perf_counter()
    rows = []
    for k, axis in enumerate(AXIS_NAMES, start=1):
        variant = base_variant(axis)
        qc = generate_with_qc(gen, orig, axis, variant, seed=42 + k)
        if not qc.discarded:
            qc.image.save(args.out / f"{args.generator}_{k}_{axis}.jpg", quality=90)
        rows.append({"id": f"R1-{k}", "ronda": 1, "eje": axis, "variante": variant,
                     "decision": "aceptada" if k % 2 else "rechazada", "motivo": None,
                     "dino": None if qc.discarded else round(qc.dino, 3), "intentos": qc.attempts,
                     "segundos": round(qc.seconds, 1)})
        print(f"  {axis}: {rows[-1]['segundos']} s, DINO {rows[-1]['dino']}", flush=True)
    report["segundos_ronda"] = round(time.perf_counter() - t0, 1)
    try:
        import torch

        if torch.cuda.is_available():
            report["vram_pico_gb"] = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
    except ImportError:
        pass
    report["variaciones"] = rows

    log = [{k: r[k] for k in ("id", "ronda", "eje", "variante", "decision", "motivo")} for r in rows]
    exp = explain(log, {}, get_llm(args.llm))
    report["agente"] = {"llm": exp.llm_name, "fuente": exp.source, "intentos": exp.attempts,
                        "segundos": round(exp.seconds, 1), "errores": exp.errors, "resumen": exp.output["resumen"]}

    (args.out / "smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

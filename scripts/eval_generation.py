"""Evaluación cuantitativa de la generación: FLUX.2 [klein] vs SD 1.5 + ControlNet Depth.

Protocolo: 50 imágenes (10 por escena) × 5 ejes (variante base de cada eje = una "ronda").
Métricas por variación: similitud DINOv2 con el original (composición), CLIP-score de la
variación frente al original y frente a la instrucción (adherencia), intentos de QC.
Por ronda: LPIPS medio entre las cinco variaciones (diversidad) y segundos por ronda.

    python scripts/eval_generation.py --methods flux sd --images data/lhq_subset
    python scripts/eval_generation.py --methods mock --limit 5      # prueba rápida sin GPU
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paisajia import vision  # noqa: E402
from paisajia.config import AXES, AXIS_NAMES, SETTINGS, base_variant  # noqa: E402
from paisajia.generation import get_generator  # noqa: E402
from paisajia.quality import generate_with_qc  # noqa: E402


def _free_gpu() -> None:
    """Libera el generador anterior y reinicia el contador de pico de VRAM."""
    import gc

    get_generator.cache_clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _peak_vram_gb() -> float | None:
    """Pico de VRAM reservada por PyTorch desde el último _free_gpu() (incluye métricas DINOv2/CLIP/LPIPS)."""
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
    except ImportError:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["flux", "sd"], choices=["flux", "sd", "mock"])
    ap.add_argument("--images", type=Path, default=Path("data/lhq_subset"))
    ap.add_argument("--limit", type=int, default=None, help="máximo de imágenes (pruebas)")
    ap.add_argument("--no-retry", action="store_true", help="medir el modelo sin reintentos de QC")
    ap.add_argument("--out", type=Path, default=Path("results/generation"))
    ap.add_argument("--save-examples", type=int, default=3, help="guardar cuadrículas de N imágenes")
    args = ap.parse_args()

    files = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))[: args.limit]
    if not files:
        sys.exit(f"No hay imágenes en {args.images}. Ejecuta antes scripts/prepare_dataset.py")
    args.out.mkdir(parents=True, exist_ok=True)
    rows, rounds, summary = [], [], {}
    gen = None
    for method in args.methods:
        gen = None
        _free_gpu()  # descarga el método anterior: FLUX y SD nunca se cargan a la vez
        gen = get_generator(method, fallback_to_mock=False)
        for k, f in enumerate(files):
            scene = f.parent.name
            orig = Image.open(f).convert("RGB").resize((1024, 1024))
            t0 = time.perf_counter()
            outs = []
            for axis in AXIS_NAMES:
                variant = base_variant(axis)
                qc = generate_with_qc(gen, orig, axis, variant, seed=1000 + k, max_retries=0 if args.no_retry else None)
                if qc.discarded:
                    rows.append({"method": gen.name, "image": f.name, "scene": scene, "axis": axis, "variant": variant,
                                 "discarded": 1})
                    continue
                txt = vision.clip_scores(qc.image, [AXES[axis]["variants"][variant]])
                rows.append({"method": gen.name, "image": f.name, "scene": scene, "axis": axis, "variant": variant,
                             "dino": qc.dino, "dino_backend": qc.dino_backend, "clip_img": qc.clip_adherence,
                             "clip_txt": txt[0] if txt else None, "attempts": qc.attempts, "discarded": 0,
                             "seconds": qc.seconds})
                outs.append(qc.image)
            secs = time.perf_counter() - t0
            lp, backend = vision.mean_pairwise_lpips(outs)
            rounds.append({"method": gen.name, "image": f.name, "scene": scene, "lpips": lp, "lpips_backend": backend,
                           "seconds": secs})
            if k < args.save_examples:
                grid = Image.new("RGB", (256 * (len(outs) + 1), 256), "white")
                grid.paste(orig.resize((256, 256)), (0, 0))
                for j, im in enumerate(outs):
                    grid.paste(im.resize((256, 256)), (256 * (j + 1), 0))
                grid.save(args.out / f"ejemplo_{gen.name}_{f.stem}.jpg", quality=90)
            print(f"\r{gen.name}: {k + 1}/{len(files)}", end="", flush=True)
        print()
        r = [x for x in rows if x["method"] == gen.name and not x["discarded"]]
        rr = [x for x in rounds if x["method"] == gen.name]
        dinos = np.array([x["dino"] for x in r])
        summary[gen.name] = {
            "variaciones": len(r),
            "descartadas": sum(x["discarded"] for x in rows if x["method"] == gen.name),
            "dino_medio": float(dinos.mean()),
            "pct_dino_>=_umbral": float((dinos >= SETTINGS.thresholds.dino_min).mean()),
            "lpips_medio_por_ronda": float(np.mean([x["lpips"] for x in rr])),
            "segundos_por_ronda": float(np.mean([x["seconds"] for x in rr])),
            "intentos_medios": float(np.mean([x["attempts"] for x in r])),
            "vram_pico_gb": _peak_vram_gb(),
            "parametros": gen.params(),
            "backends": sorted({x["dino_backend"] for x in r} | {x["lpips_backend"] for x in rr}),
            "por_eje": {a: {"dino": float(np.mean([x["dino"] for x in r if x["axis"] == a])),
                            "clip_txt": (float(np.mean([x["clip_txt"] for x in r if x["axis"] == a]))
                                         if r and r[0]["clip_txt"] is not None else None)}
                        for a in AXIS_NAMES},
            "por_escena": {s: float(np.mean([x["dino"] for x in r if x["scene"] == s]))
                           for s in sorted({x["scene"] for x in r})},
        }
    with open(args.out / "variaciones.csv", "w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for row in rows for k in row})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(args.out / "rondas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rounds[0]))
        w.writeheader()
        w.writerows(rounds)
    (args.out / "resumen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

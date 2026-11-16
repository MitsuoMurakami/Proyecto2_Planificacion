"""Selección y análisis exploratorio del subconjunto de LHQ1024.

1. Lee las imágenes de un directorio local de LHQ1024 (descargado de Kaggle) y,
   si existe, su `metadata.json` con licencias.
2. Conserva solo licencias que permiten derivados (CC0, CC BY, CC BY-SA, PDM...).
3. Preprocesa (EXIF, 1024², nitidez, rostros, marcas de agua) y clasifica la escena con CLIP.
4. Elige 10 imágenes por tipo de escena (50 en total) y las copia a data/lhq_subset/<escena>/.
5. Guarda estadísticas del análisis exploratorio en results/eda.json (+ histograma si hay matplotlib).

    python scripts/prepare_dataset.py --src D:/datasets/lhq1024 --per-scene 10
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paisajia.config import DATA_DIR, SCENES  # noqa: E402
from paisajia.preprocessing import preprocess  # noqa: E402

DERIVATIVE_OK = ("cc0", "cc-by", "cc by", "public domain", "pdm", "cc-by-sa", "cc by-sa", "unsplash")
NO_DERIVS = ("nd", "noderiv", "no deriv")


def load_licenses(src: Path) -> dict[str, str]:
    meta = src / "metadata.json"
    if not meta.exists():
        return {}
    data = json.loads(meta.read_text(encoding="utf-8"))
    items = data.items() if isinstance(data, dict) else ((d.get("filename") or d.get("file") or d.get("id"), d) for d in data)
    out = {}
    for k, v in items:
        lic = v.get("license") if isinstance(v, dict) else v
        out[Path(str(k)).stem] = str(lic or "").lower()
    return out


def license_ok(lic: str | None) -> bool:
    if lic is None:
        return False
    return any(t in lic for t in DERIVATIVE_OK) and not any(t in lic for t in NO_DERIVS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", type=Path, default=DATA_DIR / "lhq_subset")
    ap.add_argument("--per-scene", type=int, default=10)
    ap.add_argument("--max-scan", type=int, default=3000, help="imágenes a examinar como máximo")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    licenses = load_licenses(args.src)
    files = sorted(p for p in args.src.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    random.Random(args.seed).shuffle(files)
    chosen: dict[str, list] = {s: [] for s in SCENES}
    eda = {"examinadas": 0, "rechazo_licencia": 0, "rechazo_preproceso": Counter(), "escenas": Counter(),
           "tamanos": Counter(), "nitidez": [], "brillo": []}
    for f in files[: args.max_scan]:
        if all(len(v) >= args.per_scene for v in chosen.values()):
            break
        eda["examinadas"] += 1
        if licenses and not license_ok(licenses.get(f.stem)):
            eda["rechazo_licencia"] += 1
            continue
        img = Image.open(f)
        eda["tamanos"][f"{img.width}x{img.height}"] += 1
        pre = preprocess(img)
        eda["nitidez"].append(pre.blur_score)
        if not pre.ok:
            for r in pre.reasons:
                eda["rechazo_preproceso"][r.split("(")[0].strip()] += 1
            continue
        eda["brillo"].append(float(np.asarray(pre.image.convert("L")).mean()))
        eda["escenas"][pre.scene] += 1
        if len(chosen[pre.scene]) < args.per_scene:
            out = args.dst / pre.scene / f"{f.stem}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            pre.image.save(out)
            chosen[pre.scene].append({"archivo": f.name, "licencia": licenses.get(f.stem, "sin metadata")})
        print(f"\r{eda['examinadas']} examinadas · " + " ".join(f"{s}:{len(v)}" for s, v in chosen.items()), end="")

    summary = {
        "examinadas": eda["examinadas"], "rechazo_licencia": eda["rechazo_licencia"],
        "rechazo_preproceso": dict(eda["rechazo_preproceso"]), "distribucion_escenas": dict(eda["escenas"]),
        "tamanos_frecuentes": dict(eda["tamanos"].most_common(5)),
        "nitidez": {"media": float(np.mean(eda["nitidez"] or [0])), "p10": float(np.percentile(eda["nitidez"] or [0], 10))},
        "brillo_medio": float(np.mean(eda["brillo"] or [0])),
        "seleccion": chosen,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/eda.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].bar(list(eda["escenas"]), list(eda["escenas"].values()), color="#4C78A8")
        ax[0].set_title("Escenas (CLIP) en las imágenes válidas")
        ax[1].hist(eda["nitidez"], bins=30, color="#72B7B2")
        ax[1].set_title("Nitidez (varianza del laplaciano)")
        fig.tight_layout()
        fig.savefig("results/eda.png", dpi=120)
    except ImportError:
        pass
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "seleccion"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

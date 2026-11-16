"""Descarga todos los modelos de Hugging Face y los pesos de LPIPS a la caché local.

Pensado para clústeres cuyos nodos de cómputo no tienen internet (Khipu): se ejecuta
una vez en el nodo de acceso y luego los trabajos corren con HF_HUB_OFFLINE=1.
Solo descarga archivos: no carga nada en GPU ni ejecuta modelos.

    python scripts/download_models.py              # todo (~35 GB)
    python scripts/download_models.py --only flux  # flux | sd | vision
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paisajia.config import SETTINGS  # noqa: E402

# Formatos que no usamos (TensorFlow, Flax, ONNX, checkpoints sueltos de SD): se omiten para ahorrar disco.
IGNORE = ["*.h5", "*.msgpack", "*.onnx", "*.ot", "*tf_model*", "*flax_model*", "*.ckpt", "*.gguf",
          "v1-5-pruned*", "*.fp16.*", "*.non_ema.*"]

GROUPS = {
    "flux": [SETTINGS.flux_model_id],
    "sd": [SETTINGS.sd_model_id, SETTINGS.controlnet_id, "Intel/dpt-hybrid-midas"],
    "vision": [SETTINGS.dino_id, SETTINGS.clip_id],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    for group in args.only:
        for repo in GROUPS[group]:
            print(f"→ {repo}", flush=True)
            try:
                path = snapshot_download(repo, ignore_patterns=IGNORE)
                print(f"  ok: {path}")
            except Exception as exc:  # repos con licencia aceptada requieren HF_TOKEN
                print(f"  ERROR: {exc}\n  Si el modelo pide aceptar su licencia, hazlo en huggingface.co y exporta HF_TOKEN.")

    if "vision" in args.only:
        # LPIPS descarga los pesos de AlexNet (torchvision) la primera vez que se instancia.
        print("→ LPIPS (AlexNet)", flush=True)
        try:
            import lpips

            lpips.LPIPS(net="alex", verbose=False)
            print("  ok")
        except Exception as exc:
            print(f"  ERROR: {exc}")


if __name__ == "__main__":
    main()

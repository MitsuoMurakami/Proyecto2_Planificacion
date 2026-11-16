#!/bin/bash
# Instala PaisajIA en Google Colab (GPU T4). Lo llama el notebook notebooks/PaisajIA_Colab_T4.ipynb:
#     !bash scripts/colab/setup.sh            # dependencias de Python
#     !bash scripts/colab/setup.sh --ollama   # además Ollama + Qwen 2.5 3B (agente local, opcional)
# Conserva el torch de Colab (ya trae CUDA): reinstalarlo rompe el entorno y tarda varios minutos.
set -eo pipefail
cd "$(dirname "$0")/../.."

echo "== Dependencias de Python"
pip install -q -r requirements-colab.txt
if ! python -c "import diffusers, sys; sys.exit(0 if hasattr(diffusers, 'Flux2KleinPipeline') else 1)"; then
    echo "diffusers sin Flux2KleinPipeline → se instala desde la rama main"
    pip install -q "git+https://github.com/huggingface/diffusers"
fi
python - <<'PY'
import torch, diffusers, transformers
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) · diffusers {diffusers.__version__} · transformers {transformers.__version__}")
try:
    import bitsandbytes
    print(f"bitsandbytes {bitsandbytes.__version__}")
except Exception as exc:
    print(f"AVISO: bitsandbytes no disponible ({exc}); FLUX usará descarga a CPU y puede agotar la RAM de Colab")
PY

if [ "$1" = "--ollama" ]; then
    echo "== Ollama + ${PAISAJIA_OLLAMA_MODEL:-qwen2.5:3b}"
    command -v zstd > /dev/null || (apt-get -qq update && apt-get -qq install -y zstd > /dev/null)
    command -v ollama > /dev/null || curl -fsSL https://ollama.com/install.sh | sh
    mkdir -p logs
    if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
        nohup ollama serve > logs/ollama.log 2>&1 &
        for _ in $(seq 1 60); do curl -s http://127.0.0.1:11434/api/tags > /dev/null && break; sleep 1; done
    fi
    ollama pull "${PAISAJIA_OLLAMA_MODEL:-qwen2.5:3b}"
fi
echo "Instalación completa."

#!/bin/bash
# Instalación única de PaisajIA en Khipu. Ejecutar EN EL NODO DE ACCESO (único con internet):
#     bash ~/paisajia/scripts/khipu/setup.sh
# Solo instala paquetes y descarga archivos: no ejecuta modelos (eso va siempre por SLURM).
# Opcional: export HF_TOKEN=... si algún modelo de Hugging Face pide aceptar su licencia.
export PAISAJIA_HOME="${PAISAJIA_HOME:-$HOME/paisajia}"
source "$PAISAJIA_HOME/scripts/khipu/env.sh"   # antes de `set -u`: los `module load` no lo toleran
set -eo pipefail

echo "== 1/4 Entorno de Python"
python3 -c "import sys; assert sys.version_info >= (3, 10), sys.version" \
    || { echo "ERROR: se necesita Python ≥ 3.10 (module load python3/3.11.11)"; exit 1; }
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-gpu.txt
if ! python -c "import diffusers, sys; sys.exit(0 if hasattr(diffusers, 'Flux2KleinPipeline') else 1)"; then
    echo "diffusers sin Flux2KleinPipeline → se instala desde la rama main"
    pip install "git+https://github.com/huggingface/diffusers"
fi

echo "== 2/4 Ollama (en ~/ollama, sin sudo)"
if ! command -v ollama > /dev/null; then
    mkdir -p "$HOME/ollama"
    cd "$HOME/ollama"
    if curl -fL https://ollama.com/download/ollama-linux-amd64.tgz -o ollama.tgz; then
        tar -xzf ollama.tgz && rm ollama.tgz
    elif curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst -o ollama.tar.zst; then
        zstd -dc ollama.tar.zst | tar -x && rm ollama.tar.zst
    else
        echo "ERROR: no se pudo descargar Ollama; revisa el nombre del archivo en https://ollama.com/download/linux"
        exit 1
    fi
    cd "$PAISAJIA_HOME"
fi
ollama --version || true

echo "== 3/4 Qwen 2.5 3B (servidor temporal solo para la descarga)"
start_ollama 11500
ollama pull "$PAISAJIA_OLLAMA_MODEL"
kill "$OLLAMA_PID" 2>/dev/null || true

echo "== 4/4 Modelos de Hugging Face y LPIPS (~35 GB)"
python scripts/download_models.py

echo
echo "Instalación completa. Espacio usado:"
du -sh "$HF_HOME" "$OLLAMA_MODELS" "$PAISAJIA_HOME/.venv" 2>/dev/null || true
echo "Siguiente paso:  sbatch scripts/khipu/smoke.sbatch"

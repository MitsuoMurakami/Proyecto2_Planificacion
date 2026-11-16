# Entorno común de PaisajIA en Khipu. Se carga con:  source scripts/khipu/env.sh
# Todas las rutas viven en tu carpeta personal; cámbialas aquí si usas otro directorio con más cuota.

export PAISAJIA_HOME="${PAISAJIA_HOME:-$HOME/paisajia}"
export PAISAJIA_DATA_DIR="$PAISAJIA_HOME/data"

# Cachés de modelos (descargados en el nodo de acceso, leídos sin internet en los nodos de cómputo)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/.ollama/models}"
export PATH="$HOME/ollama/bin:$PATH"

# Agente local (Qwen 2.5 3B vía Ollama) y generador por defecto
export PAISAJIA_LLM=ollama
export PAISAJIA_OLLAMA_MODEL="${PAISAJIA_OLLAMA_MODEL:-qwen2.5:3b}"
export GRADIO_ANALYTICS_ENABLED=False
export PYTHONIOENCODING=utf-8

module load python3/3.11.11 2>/dev/null || module load miniconda/3.0 2>/dev/null || true
module load cuda/12.8 2>/dev/null || true

if [ -f "$PAISAJIA_HOME/.venv/bin/activate" ]; then
    source "$PAISAJIA_HOME/.venv/bin/activate"
fi
cd "$PAISAJIA_HOME" || { echo "No existe $PAISAJIA_HOME (¿subiste el proyecto?)"; return 1 2>/dev/null || exit 1; }

# Arranca Ollama en un puerto propio del trabajo (evita choques con otros usuarios del mismo nodo).
start_ollama() {
    local port="${1:-$((20000 + ${SLURM_JOB_ID:-0} % 20000))}"
    export OLLAMA_HOST="127.0.0.1:$port"
    ollama serve > "$PAISAJIA_HOME/logs/ollama-${SLURM_JOB_ID:-local}.log" 2>&1 &
    OLLAMA_PID=$!
    trap 'kill $OLLAMA_PID 2>/dev/null' EXIT
    for _ in $(seq 1 60); do
        curl -s "http://$OLLAMA_HOST/api/tags" > /dev/null && { echo "Ollama listo en $OLLAMA_HOST"; return 0; }
        sleep 1
    done
    echo "ERROR: Ollama no arrancó (ver logs/ollama-${SLURM_JOB_ID:-local}.log)"
    return 1
}

# Perfil para GPUs pequeñas (T4 de 16 GB): Qwen en CPU, FLUX a 768 px, fp16.
apply_gpu_profile() {
    local mem
    mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) · ${mem:-?} MiB"
    if [ -n "$mem" ] && [ "$mem" -lt 20000 ]; then
        echo "GPU < 20 GB → perfil T4: Qwen en CPU, generación a 768 px"
        export PAISAJIA_OLLAMA_NUM_GPU=0
        export PAISAJIA_GEN_SIZE="${PAISAJIA_GEN_SIZE:-768}"
    fi
}

mkdir -p "$PAISAJIA_HOME/logs"

#!/bin/bash
# Evaluación completa del informe en Khipu:
#   1) FLUX y 2) SD sobre las 50 imágenes de LHQ (cada método en su propio proceso: nunca juntos en GPU)
#   3) agente Qwen en 100 sesiones simuladas, con y sin verificador
#   4) planificador (Thompson vs aleatorio, solo CPU)
# Requiere data/lhq_subset/ (scripts/prepare_dataset.py).
#     sbatch scripts/khipu/eval.sbatch
#SBATCH --job-name=paisajia-eval
#SBATCH --partition=gpu
#SBATCH --nodelist=g002
#SBATCH --gres=shard:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/eval-%j.out

source "${PAISAJIA_HOME:-$HOME/paisajia}/scripts/khipu/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
apply_gpu_profile
set -x

python scripts/eval_generation.py --methods flux --images data/lhq_subset --out results/generation_flux
python scripts/eval_generation.py --methods sd   --images data/lhq_subset --out results/generation_sd

# El agente se evalúa después de la generación, con la GPU libre (Qwen puede usarla aunque sea una T4).
unset PAISAJIA_OLLAMA_NUM_GPU
start_ollama || exit 1
python scripts/eval_agent.py --llm ollama --n 100

python scripts/eval_planner.py
echo "Resultados en $PAISAJIA_HOME/results/  (Khipu no tiene backups: descárgalos)"

#!/bin/bash
# Prueba rápida (≤ 30 min) en la T4 de debug-gpu: una ronda con FLUX + un resumen con Qwen.
# Reporta tiempo por imagen, pico de VRAM y si el agente pasó el verificador.
#     sbatch scripts/khipu/smoke.sbatch
#     sbatch --export=ALL,GENERATOR=sd scripts/khipu/smoke.sbatch
#SBATCH --job-name=paisajia-smoke
#SBATCH --partition=debug-gpu
#SBATCH --gres=shard:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/smoke-%j.out

source "${PAISAJIA_HOME:-$HOME/paisajia}/scripts/khipu/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
apply_gpu_profile
start_ollama || exit 1

python scripts/smoke_test.py --generator "${GENERATOR:-flux}" --llm ollama --out "results/smoke-${SLURM_JOB_ID}"

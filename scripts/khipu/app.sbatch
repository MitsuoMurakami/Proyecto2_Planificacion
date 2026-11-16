#!/bin/bash
# Lanza la interfaz de PaisajIA en un nodo GPU de Khipu (FLUX + Qwen local).
#     sbatch scripts/khipu/app.sbatch
#     sbatch --export=ALL,GENERATOR=sd scripts/khipu/app.sbatch          # línea base
#     sbatch --nodelist=ag001 scripts/khipu/app.sbatch                   # A100 en vez de A6000
# Luego:  cat logs/app-<id>.out   → copia el comando de túnel SSH que aparece ahí.
#
# Por defecto pide g002 (RTX A6000, 48 GB): FLUX + métricas + Qwen (~18-20 GB) caben con margen.
# En la T4 (g001) se aplica solo el perfil reducido (Qwen en CPU, 768 px).
#SBATCH --job-name=paisajia-app
#SBATCH --partition=gpu
#SBATCH --nodelist=g002
#SBATCH --gres=shard:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/app-%j.out

source "${PAISAJIA_HOME:-$HOME/paisajia}/scripts/khipu/env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
apply_gpu_profile
start_ollama || exit 1

PORT=$((30000 + SLURM_JOB_ID % 20000))
NODE=$(hostname -s)
cat <<EOF

================================================================
 PaisajIA en $NODE:$PORT   (trabajo $SLURM_JOB_ID)

 En tu laptop (PowerShell), abre el túnel y déjalo abierto:
     ssh -N -L 7860:$NODE:$PORT $USER@khipu.utec.edu.pe

 Luego abre en el navegador:  http://localhost:7860
 Para terminar:  scancel $SLURM_JOB_ID
================================================================

EOF

python app.py --generator "${GENERATOR:-flux}" --llm ollama --host 0.0.0.0 --port "$PORT"

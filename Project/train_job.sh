#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=myjob.out
#SBATCH --error=myjob.err

# Module laden
module load gcc/9.4.0-pe5.34
module load USS/2022
module load miniconda3/4.12.0

# Change to working directory
cd /cfs/earth/scratch/hessluc1/ADL/

# ---- Debug Info ----
echo "Python executable:"
/cfs/earth/scratch/hessluc1/.conda/envs/project-env/bin/python --version

echo "GPU Info:"
nvidia-smi

echo "Working directory:"
pwd

# ---- Training starten ----
/cfs/earth/scratch/hessluc1/.conda/envs/project-env/bin/python \
/cfs/earth/scratch/hessluc1/ADL/ModelAnnotation.py

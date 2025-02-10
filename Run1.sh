#!/bin/sh
#SBATCH --account=stats
#SBATCH --partition=ada
#SBATCH --nodes=1 --ntasks=1
#SBATCH --job-name="Anomaly Tuning"
#SBATCH --mail-user=blnedw003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

# Your science stuff goes here...

module load python/miniconda3-py310
source activate myenv
python /home/blnedw003/Anomaly/src/hyper.py
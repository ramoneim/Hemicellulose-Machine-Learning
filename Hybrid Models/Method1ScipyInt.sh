#!/bin/bash
#SBATCH --gres=gpu:1 # Request GPU "generic resource"
#SBATCH --cpus-per-task=6 # Maximum CPY cores per GPU request: 6 on Cedar, 16 on another one
#SBATCH --mem=32000M 
#SBATCH --time=5-05:00 
#SBATCH --output=Method1ScipyInt-%N-%j.out
#SBATCH --mail-user=ranamoneim@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --account=def-yankai

module load python/3.7
module load nixpkgs/16.09
module load gcc/5.4.0
module load gcc/7.3.0
module load ipopt
module load python/3.6
module load scipy-stack
module load cuda cudnn 
source ./venv/bin/activate


python ./Method1ScipyInt.py





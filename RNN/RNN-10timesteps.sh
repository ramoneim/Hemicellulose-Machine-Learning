#!/bin/bash
#SBATCH --gres=gpu:1 # Request GPU "generic resource"
#SBATCH --cpus-per-task=6 # Maximum CPY cores per GPU request: 6 on Cedar, 16 on another one
#SBATCH --mem=32000M # Memory per node
#SBATCH --time=0-5:00 # Time (DD-HH:MM)
#SBATCH --output=RNN-10timesteps-%N-%j.out # %N for node name, %j for JobID
#SBATCH --mail-user=ranamoneim@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --account=def-yankai

module load python/3.6
module load scipy-stack
module load cuda cudnn 
source ./venv/bin/activate

python ./RNN-10timesteps.py




#!/bin/sh
#SBATCH --job-name=mpc-encoder-8g
#SBATCH --ntasks=20 # Run on a single CPU
#SBATCH --time=25:00:00 # Time limit hrs:min:sec
#SBATCH --output=mpc-encoder-8g%j.out # Standard output and error log
#SBATCH --error=mpc-encoder-8g%j.err
#SBATCH --gres=gpu:6
#SBATCH --partition=cl2_all_8G

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mbbl

for seed in 82505; do
    for env_type in "HalfCheetah-v3"; do
        #python3 mbbl/PPO-encoder/main.py --env_id ${env_type} --batch_size 8000 --max_iter 1000 --seed ${seed}
        #python3 mbbl/PPO-vanilla/main.py --env_id ${env_type} --batch_size 8000 --max_iter 1000 --seed ${seed}
        #python3 mbbl/TD3-encoder/main.py --env_id ${env_type} --max_iter 2000 --seed ${seed} &
        python3 /scratch/soumyars/mbbl/mbbl/TD3-MPC-encoder/main.py --env_id ${env_type} --max_iter 1000 --seed ${seed}
        #python3 mbbl/TD3-vanilla/main.py --env_id ${env_type} --max_iter 2000 --seed ${seed}
    done
done


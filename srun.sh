srun -n1  --gres=gpu:1 --mem=32G --ntasks-per-node=1 --job-name=ddpm \
python generate.py

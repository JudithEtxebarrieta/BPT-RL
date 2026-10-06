source ~/miniconda3/etc/profile.d/conda.sh
conda create --prefix ./env_bpt_rl python=3.10.12
conda activate ./env_bpt_rl

pip install -r requirements.txt
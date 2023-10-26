export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/data/czl/anaconda3/envs/MSTH/lib/"
export PROFILING=1
export PYTHONPATH="$PYTHONPATH:/data/czl/nerf/MSTH"
CUDA_VISIBLE_DEVICES=$1 \
    python -m scripts.train nerfacto_profiling \
    --experiment-name nerfacto_profiling \
    --vis tensorboard \
    --output-dir tmp

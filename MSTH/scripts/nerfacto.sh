export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/data/czl/anaconda3/envs/MSTH/lib/"
export PYTHONPATH="$PYTHONPATH:/data/czl/nerf/MSTH"
CUDA_VISIBLE_DEVICES=$1 \
    python -m scripts.train nerfacto_profiling \
    --experiment-name nerfacto \
    --vis wandb \
    --output-dir tmp \
    --pipeline.model.predict_normals False

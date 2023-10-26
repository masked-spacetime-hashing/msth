export PYTHONPATH="$PYTHONPATH:/opt/czl/nerf/MSTH"
CUDA_VISIBLE_DEVICES=$1 \
    python -m scripts.video_train stream-nerfacto-baseline \
    --experiment-name streamable-nerfacto \
    --vis wandb \
    --output-dir tmp

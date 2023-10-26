export PYTHONPATH="$PYTHONPATH:/opt/czl/nerf/MSTH"
CUDA_VISIBLE_DEVICES=$1 \
    python scripts/video_train.py video-tensorf-baseline \
    --vis wandb \
    --output-dir tmp

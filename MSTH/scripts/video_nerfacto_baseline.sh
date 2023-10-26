export PYTHONPATH="$PYTHONPATH:/opt/czl/nerf/MSTH"
CUDA_VISIBLE_DEVICES=$1 \
    python -m scripts.video_train video-nerfacto-baseline \
    --vis tensorboard \
    --output-dir tmp

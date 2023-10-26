viewer=${2:-'tensorboard'}
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/data/czl/anaconda3/envs/MSTH/lib/"
export PYTHONPATH="$PYTHONPATH:/data/czl/nerf/MSTH"
CUDA_VISIBLE_DEVICES=$1 \
    python -m scripts.train sth \
    --experiment-name sth \
    --vis $viewer \
    --output-dir tmp

viewer=${3:-'wandb'}
port=${4:-'7007'}
cd /opt/czl/nerf/exp/MSTH
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/data/czl/anaconda3/envs/MSTH/lib/"
# export PYTHONPATH="$PYTHONPATH:/data/czl/nerf/MSTH_new"
CUDA_VISIBLE_DEVICES=$1 \
    python -m MSTH.scripts.train ${2} \
    --experiment-name ${2} \
    --vis $viewer \
    --output-dir tmp \
    --viewer.websocket-port $port

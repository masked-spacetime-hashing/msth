viewer=${2:-'tensorboard'}
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/data/czl/anaconda3/envs/MSTH/lib/"
export PYTHONPATH="$PYTHONPATH:/data/czl/nerf/MSTH"
CUDA_VISIBLE_DEVICES=$1 \
    python -m scripts.train sth_rect \
    --experiment-name sth_rect \
    --vis $viewer \
    --output-dir tmp \
    --pipeline.datamanager.dataparser.scene_scale 4 \
    --pipeline.model.use_proposal_weight_anneal True

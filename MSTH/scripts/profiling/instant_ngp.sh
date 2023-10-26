cd /data/czl/nerf/MSTH_new
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/data/czl/anaconda3/envs/MSTH/lib/"
export PYTHONPATH="$PYTHONPATH:/data/czl/nerf/MSTH_new"
export PROFILING=1
export CSV_PATH=instant-ngp.csv
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=$1 \
    python -m MSTH.scripts.train instant-ngp \
    --experiment-name instant-ngp \
    --vis tensorboard \
    --output-dir tmp \
    --max_num_iterations 1000 \
    --steps_per_eval_image 100 \
    --data /data/machine/data/flame_salmon_image

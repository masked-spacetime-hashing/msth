cd /data/czl/nerf/MSTH_new/MSTH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/data/czl/anaconda3/envs/MSTH/lib/"
export PYTHONPATH="$PYTHONPATH:/data/czl/nerf/MSTH_new"
CUDA_VISIBLE_DEVICES=$1 \
    kernprof -lv -o line_profiler.prof scripts/train.py tp29_hierarchicay_16384_nostatic \
    --experiment-name profiling \
    --vis tensorboard \
    --max_num_iterations 500 \
    --steps_per_eval_batch 100 \
    --steps_per_eval_image 100 \
    --output-dir tmp

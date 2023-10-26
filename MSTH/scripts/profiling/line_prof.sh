cd /data/czl/nerf/MSTH_new/MSTH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/data/czl/anaconda3/envs/MSTH/lib/"
export PYTHONPATH="$PYTHONPATH:/data/czl/nerf/MSTH_new"
CUDA_VISIBLE_DEVICES=$1 \
    kernprof -lv -o line_profiler.prof scripts/train.py dsth_with_base \
    --experiment-name dsth_with_base \
    --vis tensorboard \
    --max_num_iterations 1000 \
    --output-dir tmp \
    --pipeline.datamanager.dataparser.scale_factor 0.5 \
    --pipeline.datamanager.use_stratified_pixel_sampler True \
    --pipeline.datamanager.static_dynamic_sampling_ratio 50.0 \
    --pipeline.datamanager.static_dynamic_sampling_ratio_end 10.0 \
    --pipeline.datamanager.static_ratio_decay_total_steps 20000 \
    --save_eval_video False \
    --steps_full_video 1000000000000000

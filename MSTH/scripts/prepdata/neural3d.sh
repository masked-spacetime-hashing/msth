SCENE=$1
echo "downsample"
python MSTH/scripts/prepdata/downsample.py --height 1014 -w 1352 -i /data/machine/data/$SCENE/videos -o /data/machine/data/$SCENE/videos_2/ -t area

echo "running json from pose_bound.npy"
python MSTH/scripts/prepdata/llff2nerf.py /data/machine/data/$SCENE/ --videos videos --downscale 1 --hold_list 0 --num_frames 300
python MSTH/scripts/prepdata/llff2nerf.py /data/machine/data/$SCENE/ --videos videos_2 --downscale 1 --hold_list 0 --num_frames 300

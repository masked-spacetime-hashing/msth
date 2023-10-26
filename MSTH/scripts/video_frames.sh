#!/bin/bash

input_dir=$1
output_dir=$2
frame_num=$3

# output direction
mkdir -p $output_dir

# tranverse through all video files
for video_file in $input_dir/*.{mp4,avi,mkv,flv,wmv}
do
    # 从视频文件名中提取文件名和扩展名
    filename=$(basename -- "$video_file")
    extension="${filename##*.}"
    filename="${filename%.*}"
    
    # 指定输出截图的文件名和路径
    output_file="$output_dir/$filename-$frame_num.jpg"
    
    # 使用FFmpeg截取视频的第K帧，并将其保存为JPEG文件
    ffmpeg -i "$video_file" -vf "select=eq(n\,$frame_num)" -q:v 1 "$output_file"
done

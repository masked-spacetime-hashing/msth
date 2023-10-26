import subprocess
import os
import time


def get_gpu_memory_usage():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"], stdout=subprocess.PIPE
    )
    output = result.stdout.decode("utf-8").strip()
    output = output.split("\n")
    infos = []
    for gpu_id, content in enumerate(output):
        gpu_memory = [float(x.strip()) for x in content.split(",")]
        infos.append({"used": gpu_memory[0], "total": gpu_memory[1]})
    return infos


print(get_gpu_memory_usage())
with open("/data/czl/nerf/MSTH_new/MSTH/scripts/task_4_19.txt", "r") as f:
    tasks = f.readlines()

using_gpu_ids = [2, 3]

tasks = [task.strip() for task in tasks]
while len(tasks) > 0:
    infos = get_gpu_memory_usage()
    for gpu_id in using_gpu_ids:
        if infos[gpu_id]["used"] < 100:
            cmd = f"bash /data/czl/nerf/MSTH_new/MSTH/scripts/run.sh {gpu_id} {tasks.pop(0)} wandb &"
            print(f"GPU {gpu_id} is available, Running: {cmd}")
            os.system(cmd)
            time.sleep(50)
            break

    # os.system(cmd)
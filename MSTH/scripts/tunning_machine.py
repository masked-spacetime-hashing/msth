from MSTH.configs.method_configs import *
import numpy as np
import itertools
import random
import os
from pathlib import Path

base_method = method_configs["05-horse-10-2-hidden-dim-128"]


def setp(exps):
    def setfunc(x, v):
        if isinstance(exps, (tuple, list)):
            for exp in exps:
                command = "x." + exp + "=v"
                exec(command)
        else:
            command = "x." + exps + "=v"
            exec(command)

    return setfunc


set_functions = {
    "dataset": setp("pipeline.datamanager.dataparser.data"),
}

potential_values = {
    "dataset": [
        Path("/data/machine/data/immersive/05_Horse_2"),
        Path("/data/machine/data/immersive/01_Welder_2"),
        Path("/data/machine/data/immersive/09_Alexa_Meade_Exhibit_2"),
        Path("/data/machine/data/immersive/10_Face_2"),
        Path("/data/machine/data/immersive/02_Flames_2"),
        Path("/data/machine/data/immersive/11_Alexa_2"),
        Path("/data/machine/data/immersive/04_Truck_2"),
    ]
}

all_hyper_parameter_key = potential_values.keys()
all_hyper_parameter_value = [potential_values[k] for k in all_hyper_parameter_key]
all_specs = list(itertools.product(*all_hyper_parameter_value))
all_specs = [{k: v for k, v in zip(all_hyper_parameter_key, spec)} for spec in all_specs]
random.shuffle(all_specs)
print("==== ALL SPECS ====")
print(all_specs)

for i, spec in enumerate(all_specs):
    method_configs[f"imm_{i}_4_23"] = copy.deepcopy(base_method)
    for k, v in spec.items():
        set_functions[k](method_configs[f"imm_{i}_4_23"], v)
    print(method_configs[f"imm_{i}_4_23"])

emport math
import numpy as np

T = 2 ** 19
base_res = 16
max_res = 2048
num_levels = 16
growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))

R = base_res
for level in range(num_levels):
    a = [0 for i in range(T)]
    for i in range(R):
        for j in range(R):
            for k in range(R):
                hash_value = (i * 1) ^ (j * 2654435761) ^ (k * 805459861)
                hash_value = hash_value % T
                a[hash_value] += 1

    print(T)
    print(R)
    a = np.array(a)
    print(f"collision rate: {(a>1).sum()/T}")
    R = int(R * growth_factor)

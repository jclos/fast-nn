import numpy as np
import random
import time
import vectorised_nnsearch as vnn


def generate_dataset(n_features, n_rows):
    return np.random.rand(n_rows, n_features)


k = 5
n_instances = 5000
n_features = 300
n_tests = 5

times_vec_cos = []
times_leg = []
times_vec = []

size_start = 1000
size_end = 20000
steps = 10

size_step = (size_end - size_start)/steps

results = np.empty((steps,4))

x_axis = [int(x) for x in np.arange(size_start, size_end, size_step)]

print(x_axis)

for ds in range(0, steps):
    data_size = int(size_start + size_step * ds)
    print("Data size: ", data_size)
    results[ds,0] = data_size
    for i in range(0, n_tests):
        dataset = generate_dataset(n_features, data_size)
        new_instance = generate_dataset(n_features, 1)
        order = [0, 1, 2]  # to make sure there isn't some sort of caching fudging the results
        random.shuffle(order)
        for j in range(0, 3):
            d = order[j]
            if d == 0:
                start = time.time()
                nn5 = vnn.get_neighbours_cosine(new_instance, dataset, k)
                end = time.time()
                duration = end - start
                times_vec_cos.append(duration)
            if d == 1:
                start = time.time()
                nn5 = vnn.get_neighbours_legacy(new_instance, dataset, k)
                end = time.time()
                duration = end - start
                times_leg.append(duration)
            if d == 2:
                start = time.time()
                nn5 = vnn.get_neighbours(new_instance, dataset, k)
                end = time.time()
                duration = end - start
                times_vec.append(duration)
    results[ds,1] = np.mean(times_leg)
    results[ds,2] = np.mean(times_vec)
    results[ds,3] = np.mean(times_vec_cos)

print(results)

np.savetxt("results.csv", results, delimiter=",")

# import matplotlib
# import matplotlib.pyplot as plt
#
# plt.plot(results)
# plt.axis(x_axis)
# plt.show()


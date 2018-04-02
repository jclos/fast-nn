import numpy as np
import random
import time
import vectorised_nnsearch as vnn


def generate_dataset(n_features, n_rows):
    return np.random.rand(n_rows, n_features)

k = 5
n_instances = 10000
n_features = 300

n_tests = 10

times_vec_cos = []
times_leg = []
times_vec = []

for i in range(0, n_tests):
    dataset = generate_dataset(n_features, n_instances)
    new_instance = generate_dataset(n_features, 1)
    order = [0, 1, 2] # to make sure there isn't some sort of caching fudging the results
    random.shuffle(order)

    for j in range(0,3):
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


print("Vectorised cosine")
print(np.mean(times_vec_cos))
print(np.std(times_vec_cos))
print("Averaged over ", len(times_vec_cos), " elements")

print("Legacy")
print(np.mean(times_leg))
print(np.std(times_leg))
print("Averaged over ", len(times_leg), " elements")

print("Vectorised distance")
print(np.mean(times_vec))
print(np.std(times_vec))
print("Averaged over ", len(times_vec), " elements")
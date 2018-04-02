import numpy as np
import operator
import time


def get_euclidean_distance(instance1, instance2):
    '''
    Calculates euclidean distance between two instances of data
    '''
    #    distance = 0
    #    for x in range(length):
    #        distance += pow((instance1[x] - instance2[x]), 2)
    return np.linalg.norm(np.array(instance1) - np.array(instance2))


def get_neighbours_cosine(instance, dataset, n):
    return np.argsort(np.dot(instance, dataset.T) / (np.linalg.norm(dataset, axis=1) * np.linalg.norm(instance)))[n:]

def get_neighbours(instance, dataset, n):
    return dataset[np.argsort(np.linalg.norm(dataset - instance, axis=1))[:n]]

def get_neighbours_legacy(test_instance, training_set, k):
    distances = []
    for x in range(len(training_set)):
        dist = get_euclidean_distance(test_instance, training_set[x])
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for x in range(k):
        neighbours.append(distances[x][0])
    return neighbours


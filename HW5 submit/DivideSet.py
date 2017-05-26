import numpy as np


def divide(data, c, value):

    list1 = [data[i] for i in range(len(data)) if data[i][c] >= value]
    list2 = [data[i] for i in range(len(data)) if data[i][c] < value]

    return list1, list2


def label_count(data):

    labels = {}

    for r in data:

        label = r[-1]
        if label not in labels:
            labels[label] = 0
        labels[label] += 1

    return labels


def entropy(data):

    ent = 0.0
    labels = label_count(data)
    for label in labels.keys():
        prob = labels[label]/float(len(data))
        ent -= (prob * np.log(prob)/np.log(2))

    return ent




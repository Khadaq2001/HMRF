import numpy as np
import pandas as pd

def label_reverse(labels_list):
    labels_list = 1-labels_list 
    return labels_list


def difference(x, y):
    return np.abs(x - y)


def init_energy(labels_mtx, pixels, beta, cls_para, neighbor_indices, in_tissue):
    labels_mtx
    energy = 0.0
    rows, cols = labels_mtx.shape
    print(rows, cols)
    for i in range(rows):
        for j in range(cols):
            mean, var = cls_para[labels_mtx[i, j]]
            energy += np.log(np.sqrt(2 * np.pi * var))
            energy += (pixels[i, j] - mean) ** 2 / (2 * var)
            for a, b in neighbor_indices:
                a += i
                b += j
                # print(a, b)
                if (a < rows) and (b < cols) and in_tissue[a, b]:
                    energy += beta * difference(labels_mtx[i, j], labels_mtx[a, b])
    return energy


def delta_energy(labels_mtx, pixels, index, new_label, beta, cls_para, neighbor_indices, in_tissue):
    labels_mtx
    (i, j) = index
    rows, cols = labels_mtx.shape
    mean, var = cls_para[labels_mtx[i, j]]
    init_energy = np.log(np.sqrt(2 * np.pi * var)) + (pixels[i, j] - mean) ** 2 / (2 * var)
    for a, b in neighbor_indices:
        a += i
        b += j
        if (a < rows) and (b < cols) and in_tissue[a, b]:
            init_energy += beta * difference(labels_mtx[i, j], labels_mtx[a, b])
    mean_new, var_new = cls_para[new_label]
    new_energy = np.log(np.sqrt(2 * np.pi * var_new)) + (pixels[i, j] - mean_new) ** 2 / (2 * var_new)
    for a, b in neighbor_indices:
        a += i
        b += j
        if (a < rows) and (b < cols) and in_tissue[a, b]:
            new_energy += beta * difference(new_label, labels_mtx[a, b])
    # print (new_energy, init_energy)
    return new_energy - init_energy


def temp_function(method="exponential"):
    if method == "exponential":
        return lambda x: 0.99 * x


def distribution_update(new_label, cluster, cls_para):
    mean = np.mean(cluster)
    var = np.var(cluster)
    if var == 0:
        var = 1e-5
    cls_para[new_label] = (mean, var)

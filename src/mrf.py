import numpy as np
import pandas as pd
import scanpy as sc
import parmap  # type: ignore
import multiprocessing as mp
import random
from sklearn import mixture
from tqdm import tqdm


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


def annealing(
    labels_mtx: np.ndarray,
    cls,
    cls_para,
    pixels,
    beta,
    temp_function,
    neighbor_indices,
    in_tissue,
    max_iteration=10000,
    initial_temp=1000,
):
    (rows, cols) = labels_mtx.shape
    current_energy = init_energy(labels_mtx, pixels, beta, cls_para, neighbor_indices, in_tissue)
    current_tmp = initial_temp
    total_change = 0
    iter = 0
    for i in tqdm(range(int(max_iteration))):
        changed = 0
        i = random.randint(1, rows - 1)
        j = random.randint(1, cols - 1)
        new_list = list(cls)
        new_list.remove(labels_mtx[i, j])
        new_label = random.choice(new_list)
        delta = delta_energy(
            labels_mtx,
            pixels,
            (i, j),
            new_label,
            beta,
            cls_para,
            neighbor_indices,
            in_tissue,
        )

        r = random.uniform(0, 1)

        if delta < 0:
            labels_mtx[i, j] = new_label
            current_energy += delta
            changed = 1
        else:
            try:
                if -delta / current_tmp < -600:
                    k = 0
                else:
                    k = np.exp(-delta / current_tmp)
            except:
                k = 0

            if r < k:
                labels_mtx[i, j] = new_label
                current_energy += delta
                changed = 1

        if temp_function:
            current_tmp = temp_function(current_tmp)

        if changed:
            cluster = pixels[labels_mtx == new_label]
            distribution_update(new_label, cluster, cls_para)
            total_change += 1
        iter += 1
    print(f"{total_change} pixels changed after {iter} iterations")
    return labels_mtx


def icm_em_process(
    labels_mtx,
    pixels,
    beta,
    cls,
    cls_para,
    neighbor_indices,
    in_tissue,
    coord,
    exp,
    icm_iter=10,
    max_iter=100,
):
    for _ in tqdm(range(max_iter)):
        delta = float("-inf")
        # ICM step
        iter = 0
        while (iter < icm_iter) and (delta < -0.01):
            delta = 0
            changed = 0
            np.random.shuffle(coord)
            for i, j in coord:
                new_list = list(cls)
                new_list.remove(labels_mtx[i, j])
                new_label = random.choice(new_list)
                temp_delta = delta_energy(
                    labels_mtx,
                    pixels,
                    (i, j),
                    new_label,
                    beta,
                    cls_para,
                    neighbor_indices,
                    in_tissue,
                )
                print(temp_delta)
                if temp_delta < 0:
                    labels_mtx[i, j] = new_label
                    delta += temp_delta
            iter += 1
    return labels_mtx, cls_para


"""
        # E Step
        cluster_prob = np.zeros((len(coord), len(cls)))
        for i in range(len(coord)):
            x, y = coord[i]
            for k in range(len(cls)):
                mean, var = cls_para[k]
                cluster_prob[i, k] = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((pixels[x, y] - mean) ** 2) / (2 * var))
        cluster_prob /= np.sum(cluster_prob, axis=1).reshape(-1, 1)
        # print(cluster_prob.shape, exp.shape)

        # M Step
        for k in range(len(cls)):
            mean = np.sum(cluster_prob[:, k].reshape(-1, 1) * exp) / np.sum(cluster_prob[:, k])
            var = np.sum(cluster_prob[:, k].reshape(-1, 1) * (exp - mean) ** 2) / np.sum(cluster_prob[:, k])
            cls_para[k] = (mean, var)
            print(f"mean: {mean} , var : {var}")
"""


def mrf_process(
    adata,
    gene_id,
    beta,
    n_components=2,
    icm_iter=10,
    max_iter=10,
    temp_function=lambda x: 0.99 * x,
    max_iteration=10000,
    neighbor_indice=[(-1, 1), (1, 1), (1, -1), (1, 1), (0, 2), (0, -2)],
):
    """
    Marfov random field complete part
    """
    coord = adata.obs[["array_row", "array_col"]].values
    exp = adata[:, gene_id].X.toarray()
    rows, cols = np.max(coord, axis=0)
    pixels = np.zeros((rows + 1, cols + 1))
    labels_mtx = np.zeros((rows + 1, cols + 1), dtype=int)
    in_tissue = np.zeros((rows + 1, cols + 1), dtype=bool)
    labels_list = np.zeros(len(coord))
    gmm = mixture.GaussianMixture(n_components=n_components)
    gmm.fit(exp)
    pred = gmm.predict(exp)
    cls = set(pred)
    for i, (x, y) in enumerate(coord):
        pixels[x, y] = exp[i]
        labels_mtx[x, y] = pred[i]
        in_tissue[x, y] = True
    cls_para = gmm.means_.reshape(-1), gmm.covariances_.reshape(-1)
    cls_para = np.array(cls_para).T
    labels_mtx, cls_para = icm_em_process(
        labels_mtx,
        pixels,
        beta,
        cls,
        cls_para,
        neighbor_indice,
        in_tissue,
        coord,
        exp=exp,
        icm_iter=icm_iter,
        max_iter=max_iter,
    )
    for i, (x, y) in enumerate(coord):
        labels_list[i] = labels_mtx[x, y]
    return labels_list

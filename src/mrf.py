import numpy as np
import pandas as pd
import scanpy as sc
import parmap  # type: ignore
import multiprocessing as mp
import random
from sklearn import mixture
from tqdm import tqdm
from function import *


def annealing(
    labels_mtx: np.ndarray,
    cls: set,
    cls_para: np.ndarray,
    pixels: np.ndarray,
    beta: float,
    temp_function: callable,
    in_tissue: np.ndarray,
    neighbor_indices: list,
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
    icm_iter=3,
    max_iter=5,
):
    for _ in tqdm(range(max_iter)):
        # ICM step
        temp_coord = np.copy(coord)
        delta = float("-inf")
        iter = 0
        changed = 0
        while (iter < icm_iter) and (delta < -0.01):
            delta = 0
            changed = 0
            np.random.shuffle(temp_coord)
            for i, j in temp_coord:
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
                # print(temp_delta)
                if temp_delta < 0:
                    
                    labels_mtx[i, j] = new_label
                    delta += temp_delta
                    changed += 1
            iter += 1

        cluster_prob = np.zeros((len(coord), len(cls)))
        # E Step
        for i in range(len(coord)):
            x, y = coord[i]
            for k in range(len(cls)):
                mean, var = cls_para[k]
                cluster_prob[i, k] = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((pixels[x, y] - mean) ** 2) / (2 * var))
        cluster_prob /= np.sum(cluster_prob, axis=1).reshape(-1, 1)
        # M Step
        for k in range(len(cls)):
            mean = np.sum(cluster_prob[:, k].reshape(-1, 1) * exp) / np.sum(cluster_prob[:, k])
            var = np.sum(cluster_prob[:, k].reshape(-1, 1) * (exp - mean) ** 2) / np.sum(cluster_prob[:, k])
            var = 1e-5 if var == 0 else var
            cls_para[k] = (mean, var)
            # print(f"k: {k}, mean: {mean}, var : {var}")
    return labels_mtx, cls_para


def mrf_process(
    adata: sc.AnnData,
    gene_id: str,
    beta: float,
    n_components: int = 2,
    max_iteration: int = 10000,
    temp_function=lambda x: 0.99 * x,
    neighbor_indice: list = [(-1, 1), (1, 1), (1, -1), (1, 1), (0, 2), (0, -2)],
):
    print(neighbor_indice)
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
    labels_mtx = annealing(labels_mtx, cls, cls_para, pixels, beta, temp_function, in_tissue, neighbor_indice, max_iteration=max_iteration)
    print(cls_para)
    for i, (x, y) in enumerate(coord):
        labels_list[i] = labels_mtx[x, y]
    return labels_list


def mrf_with_icmem(
    adata: sc.AnnData,
    gene_id: str,
    beta: float,
    max_iter: int,
    n_components: int = 2,
    neighbor_indice=[(-1, 1), (1, 1), (1, -1), (1, 1), (0, 2), (0, -2)],
):
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
    labels_mtx, cls_para, cluster_prob = icm_em_process(
        labels_mtx,
        pixels,
        beta,
        cls,
        cls_para,
        neighbor_indice,
        in_tissue,
        coord,
        exp=exp,
        icm_iter=10,
        max_iter=max_iter,
    )
    print(cls_para)
    for i, (x, y) in enumerate(coord):
        labels_list[i] = labels_mtx[x, y]
    return labels_list, cluster_prob

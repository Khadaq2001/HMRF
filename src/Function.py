import numpy as np
import pandas as pd
import scanpy as sc
from sklearn import mixture
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp
from tqdm import tqdm
import random
from scipy import sparse as sp
from scipy.spatial.distance import cdist


def label_reverse(labels_list):
    labels_list = 1 - labels_list
    return labels_list


def difference(x, y):
    return np.abs(x - y)


def init_energy(label_list, exp, graph, cls_para, beta):
    energy = 0.0
    for i in range(graph.shape[0]):
        neighbor_indices = graph[i].indices
        mean, var = cls_para[label_list[i]]
        energy += np.log(np.sqrt(2 * np.pi * var))
        energy += (exp[i] - mean) ** 2 / (2 * var)
        for neighbor in neighbor_indices:
            energy += beta * difference(label_list[i], label_list[neighbor])
    return energy


def delta_energy(label_list, index, exp, graph, cls_para, new_label, beta):
    neighbor_indices = graph[index].indices
    mean, var = cls_para[label_list[index]]
    init_energy = np.log(np.sqrt(2 * np.pi * var)) + (exp[index] - mean) ** 2 / (
        2 * var
    )
    for neighbor in neighbor_indices:
        init_energy += beta * difference(label_list[index], label_list[neighbor])
    mean_new, var_new = cls_para[new_label]
    new_energy = np.log(np.sqrt(2 * np.pi * var_new)) + (exp[index] - mean_new) ** 2 / (
        2 * var_new
    )
    for neighbor in neighbor_indices:
        new_energy += beta * difference(new_label, label_list[neighbor])
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


def label_resort(means, label_list):
    cls_labels = np.argmax(means[:, 0])
    new_labels = np.zeros_like(label_list)
    new_labels[label_list == cls_labels] = 1
    return new_labels


def get_binary_weight(adata, radius=1):
    coor = adata.obs
    coor = coor.loc[:, ["imagerow", "imagecol"]]
    coor.index = adata.obs.index
    coor.columns = ["array_row", "array_col"]

    nbrs = NearestNeighbors(radius=radius).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(
            pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it]))
        )
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ["Cell1", "Cell2", "Distance"]
    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net["Distance"] > 0,]
    id_cell_trans = dict(
        zip(
            range(coor.shape[0]),
            np.array(coor.index),
        )
    )
    Spatial_Net["Cell1"] = Spatial_Net["Cell1"].map(id_cell_trans)
    Spatial_Net["Cell2"] = Spatial_Net["Cell2"].map(id_cell_trans)
    # out = pd.get_dummies(Spatial_Net.set_index("Cell1")['Cell2'],sparse=True).max(level=0)
    G_df = Spatial_Net
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df["Cell1"] = G_df["Cell1"].map(cells_id_tran)
    G_df["Cell2"] = G_df["Cell2"].map(cells_id_tran)
    G = sp.coo_matrix(
        (np.ones(G_df.shape[0]), (G_df["Cell1"], G_df["Cell2"])),
        shape=(adata.n_obs, adata.n_obs),
    )
    G = G + sp.eye(G.shape[0])

    return G


def getNeighborIndex(neighborMatirx):
    neighbor_indices = []
    for i in range(neighborMatirx.shape[0]):
        for j in range(neighborMatirx.shape[1]):
            if neighborMatirx[i, j] == 1:
                neighbor_indices.append((i, j))
    return neighbor_indices


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
    current_energy = init_energy(
        labels_mtx, pixels, beta, cls_para, neighbor_indices, in_tissue
    )
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
    labels_mtx = annealing(
        labels_mtx,
        cls,
        cls_para,
        pixels,
        beta,
        temp_function,
        in_tissue,
        neighbor_indice,
        max_iteration=max_iteration,
    )
    print(cls_para)
    for i, (x, y) in enumerate(coord):
        labels_list[i] = labels_mtx[x, y]
    return labels_list


def read_spatial_expression(
    file,
    sep="\s+",
    num_exp_genes=0.01,
    num_exp_spots=0.01,
    min_expression=1,
    drop=False,
):
    """
    Read raw data and returns pandas data frame of spatial gene express
    and numpy ndarray for single cell location coordinates;


    :param file: csv file for spatial gene expression;
    :rtype: coord (spatial coordinates) shape (n, 2); data: shape (n, m);
    """
    counts = pd.read_csv(file, sep=sep, index_col=0)
    print("raw data dim: {}".format(counts.shape))

    num_spots = len(counts.index)
    num_genes = len(counts.columns)
    min_genes_spot_exp = round((counts != 0).sum(axis=1).quantile(num_exp_genes))
    print(
        "Number of expressed genes a spot must have to be kept "
        "({}% of total expressed genes) {}".format(num_exp_genes, min_genes_spot_exp)
    )

    mark_points = np.where((counts != 0).sum(axis=1) < min_genes_spot_exp)[0]
    print("Marked {} spots".format(len(mark_points)))

    if len(mark_points) > 0:
        noiseInd = [counts.shape[0] - 1 - i for i in range(len(mark_points))]
        if drop == False:
            temp = [val.split("x") for val in counts.index.values]
            coord = np.array([[float(a[0]), float(a[1])] for a in temp])

            similar_points = np.argsort(cdist(coord[mark_points, :], coord), axis=1)[
                :, 1
            ]
            for i, j in zip(mark_points, similar_points):
                counts.iloc[i, :] = counts.iloc[j, :]

            mark_counts = counts.iloc[mark_points, :]
            dropped_counts = counts.drop(counts.index[mark_points])
            counts = pd.concat([dropped_counts, mark_counts])

        else:
            counts = counts[(counts != 0).sum(axis=1) >= min_genes_spot_exp]
    else:
        counts = counts
        noiseInd = []

    # Spots are columns and genes are rows
    counts = counts.transpose()
    # Remove noisy genes
    min_features_gene = round(len(counts.columns) * num_exp_spots)
    print(
        "Removing genes that are expressed in less than {} "
        "spots with a count of at least {}".format(min_features_gene, min_expression)
    )
    counts = counts[(counts >= min_expression).sum(axis=1) >= min_features_gene]
    print("Dropped {} genes".format(num_genes - len(counts.index)))

    temp = [val.split("x") for val in counts.columns.values]
    coord = np.array([[float(a[0]), float(a[1])] for a in temp])

    data = counts.transpose()

    return coord, data, noiseInd

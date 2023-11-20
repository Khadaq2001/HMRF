import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp


def label_reverse(labels_list):
    labels_list = 1 - labels_list
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


def label_resort(means, label_list):
    cls_labels = np.argmax(means[:, 0])
    new_labels = np.zeros_like(label_list)
    new_labels[label_list == cls_labels] = 1
    return new_labels


def get_binary_weight(adata, rad_cutoff):
    coor = adata.obs
    coor = coor.loc[:, ["imagerow", "imagecol"]]
    coor.index = adata.obs.index
    coor.columns = ["imagerow", "imagecol"]

    nbrs = NearestNeighbors(radius=rad_cutoff).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))
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
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df["Cell1"], G_df["Cell2"])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    return G


def get_binary_weight(adata, radius=6):
    coor = adata.obs
    coor = coor.loc[:, ["imagerow", "imagecol"]]
    coor.index = adata.obs.index
    coor.columns = ["imagerow", "imagecol"]

    nbrs = NearestNeighbors(radius = radius).fit(coor)
    distances, indices = nbrs.radius_neighbors (coor, return_distance=True)
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))
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
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df["Cell1"], G_df["Cell2"])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    return G


def getNeighborIndex(neighborMatirx):
    neighbor_indices = []
    for i in range(neighborMatirx.shape[0]):
        for j in range(neighborMatirx.shape[1]):
            if neighborMatirx[i, j] == 1:
                neighbor_indices.append((i, j))
    return neighbor_indices
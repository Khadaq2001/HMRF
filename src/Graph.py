from hmac import new
import numpy as np
import pandas as pd
import scanpy as sc
import parmap  # type: ignore
import multiprocessing as mp
import random
from sklearn import mixture, neighbors
from tqdm import tqdm
from scipy import sparse as sp


class GeneGraph:

    """
    Construct gene graph and implement HMRF in spatial transcriptomics
    """

    def __init__(self, adata: sc.AnnData, gene_id: str, radius: int = 1):
        self.exp = adata[:, gene_id].X.toarray()
        self.cellNum = adata.n_obs
        if "array_row" in adata.obs.columns:
            self.coord = adata.obs[["array_row", "array_col"]].values
        else:
            self.coord = adata.obs[["imagerow", "imagecol"]].values
        self.constructGraph(radius)

    def constructGraph(self, radius: int = 1):
        """
        Construce gene graph based on the nearest neighbor
        """
        nbrs = neighbors.NearestNeighbors(radius=radius).fit(self.coord)
        distance, indices = nbrs.radius_neighbors(self.coord, return_distance=True)
        KNN_list = []
        for i in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([i] * indices[i].shape[0], indices[i], distance[i])))
        KNN_df = pd.concat(KNN_list)
        KNN_df.columns = ["cell1", "cell2", "distance"]
        Spatial_Net = KNN_df.copy()
        Spatial_Net = Spatial_Net.loc[Spatial_Net["distance"] > 0,]
        graph = sp.coo_matrix((np.ones(Spatial_Net.shape[0]), (Spatial_Net["cell1"], Spatial_Net["cell2"])), shape=(self.cellNum, self.cellNum))
        graph = graph + sp.eye(graph.shape[0])
        self.graph = graph
        print("Graph constructed")

    def mrf_with_icmem(self, beta, n_components=2, icm_iter=10, max_iter=10):
        """
        Implement HMRF with ICMEM
        """
        gmm = mixture.GaussianMixture(n_components=n_components).fit(self.exp)
        means, covs = gmm.means_, gmm.covariances_
        pred = gmm.predict(self.exp).reshape(-1)
        cls = set(pred)
        cls_para = means.reshape(-1), covs.reshape(-1)
        cls_para = np.array(cls_para).T
        label_list = self._icmem(pred, beta, cls, cls_para, self.exp, self.graph, icm_iter, max_iter)
        print(cls_para)
        label_list = self._label_resort(means, label_list)
        self.label = label_list

    def _icmem(
        self,
        label_list: np.ndarray,
        beta: float,
        cls: set,
        cls_para: np.ndarray,
        exp: np.ndarray,
        graph: sp.csr_matrix,
        icm_iter: int = 10,
        max_iter: int = 10,
    ):
        cellNum = graph.shape[0]
        clsNum = len(cls)
        for _ in tqdm(range(max_iter)):
            # ICM step
            temp_order = np.arange(cellNum)
            delta = float("-inf")
            iter = 0
            changed = 0
            while (iter < icm_iter) and (delta < -0.01):
                delta = 0
                changed = 0
                np.random.shuffle(temp_order)
                for i in temp_order:
                    clsList = list(cls)
                    clsList.remove(label_list[i])
                    new_label = random.choice(clsList)
                    temp_delta = self._delta_energy(label_list, i, exp, graph, cls_para, new_label, beta)
                    if temp_delta < 0:
                        label_list[i] = new_label
                        delta += temp_delta
                        changed += 1
                iter += 1

            cluster_prob = np.zeros([cellNum, clsNum])
            # E Step
            for i in range(cellNum):
                for k in range(clsNum):
                    mean, var = cls_para[k]
                    cluster_prob[i, k] = (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((exp[i] - mean) ** 2) / (2 * var))
            cluster_prob /= np.sum(cluster_prob, axis=1).reshape(-1, 1)
            # M Step
            for k in range(clsNum):
                mean = np.sum(cluster_prob[:, k].reshape(-1, 1) * exp) / np.sum(cluster_prob[:, k])
                var = np.sum(cluster_prob[:, k].reshape(-1, 1) * (exp - mean) ** 2) / np.sum(cluster_prob[:, k])
                var = 1e-5 if var == 0 else var
                cls_para[k] = (mean, var)
        return label_list

    def impute(self):
        """
        Impute the expression by considering neighbor cells
        """
        self.exp = self._impute(self.exp, self.graph)
        return self.exp

    def _label_resort(self, means, label_list):  # Set the label with the highest mean as 1
        cls_labels = np.argmax(means[:, 0])
        new_labels = np.zeros_like(label_list)
        new_labels[label_list == cls_labels] = 1
        return new_labels

    def _delta_energy(self, label_list, index, exp, graph, cls_para, new_label, beta):
        neighbor_indices = graph[index].indices
        mean, var = cls_para[label_list[index]]
        init_energy = np.log(np.sqrt(2 * np.pi * var)) + (exp[index] - mean) ** 2 / (2 * var)
        for neighbor in neighbor_indices:
            init_energy += beta * self._difference(label_list[index], label_list[neighbor])
        mean_new, var_new = cls_para[new_label]
        new_energy = np.log(np.sqrt(2 * np.pi * var_new)) + (exp[index] - mean_new) ** 2 / (2 * var_new)
        for neighbor in neighbor_indices:
            new_energy += beta * self._difference(new_label, label_list[neighbor])
        # print (new_energy, init_energy)
        return new_energy - init_energy

    def _difference(self, x, y):
        return np.abs(x - y)

    def _impute(self, exp, graph):
        weightedGragh = graph.copy()
         

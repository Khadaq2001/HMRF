import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy import sparse as sp

from src.Function import delta_energy


class SingleGeneGraph:
    """
    Construct gene graph and implement HMRF in spatial transcriptomics
    """

    def __init__(
        self,
        gene_id: str,
        exp: pd.DataFrame,
        coord: np.ndarray,
        kneighbors: int,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.exp = exp.loc[:, gene_id].values
        self.cellNum = exp.shape[0]
        self.coord = coord
        self.graph = self._construct_graph(self.coord, kneighbors)
        self.corr = self._get_corr(exp, n_comp=10)

    def mrf_with_icmem(self, beta, n_components=2, icm_iter=3, max_iter=10):
        """
        Implement HMRF with ICM-EM
        """
        gmm = GaussianMixture(n_components=n_components).fit(self.exp.reshape(-1, 1))
        means, covs = gmm.means_.ravel(), gmm.covariances_.ravel()
        pred = gmm.predict(self.exp.reshape(-1, 1))
        cls = range(n_components)
        clsPara = np.column_stack((means, covs))
        labelList = self._icmem(
            pred, beta, cls, clsPara, self.exp, self.graph, icm_iter, max_iter
        )
        if self.verbose:
            print(clsPara)
        labelList = self._label_resort(means, labelList)
        self.label = labelList

    def impute(self, alpha: float = 0.5, theta: float = 0.5):
        """
        Impute the expression by considering neighbor cells

        Args:
            alpha : The scaling weight for the correlation matrix.
            theta : The replacement value for non-matching labels in the label matrix

        Returns:
            None

        """
        label = self.label
        graph = self.graph.toarray()
        corrMatrix = abs(np.multiply(graph, self.corr))
        corrMatrix = corrMatrix - np.eye(corrMatrix.shape[0])
        corrMatrix = alpha * corrMatrix / corrMatrix.sum(axis=1).reshape(-1, 1)
        adjacencyMatrix = corrMatrix + np.eye(corrMatrix.shape[0])
        labelMatrix = (label.reshape(-1, 1) == label.reshape(1, -1)).astype(float)
        labelMatrix[labelMatrix == 0] = theta
        adjacencyMatrix = np.multiply(adjacencyMatrix, labelMatrix)
        imputedExp = np.matmul(adjacencyMatrix, self.exp)
        self.imputedExp = imputedExp
        if self.verbose:
            print("Imputation finished")

    @staticmethod
    def _construct_graph(coord: np.ndarray, kneighbors: int = 6):
        """
        Construct gene graph based on the nearest neighbors
        """
        graph = (
            NearestNeighbors(n_neighbors=kneighbors).fit(coord).kneighbors_graph(coord)
        )
        return graph

    @staticmethod
    def _get_corr(exp_matrix: np.ndarray, n_comp: int = 10):
        """
        Calculate the correlation between cells based on the principal components
        """
        return np.corrcoef(
            PCA(n_comp).fit_transform(StandardScaler().fit_transform(exp_matrix))
        )

    def _label_resort(self, means, labelList):
        # Set the label with the highest mean as 1
        clsLabel = np.argmax(means)
        newLabels = np.zeros_like(labelList)
        newLabels[labelList == clsLabel] = 1
        return newLabels

    def _icmem(
        self,
        labelList: np.ndarray,
        beta: float,
        cls: set,
        clsPara: np.ndarray,
        exp: np.ndarray,
        graph: sp.csr_matrix,
        icm_iter: int = 2,
        max_iter: int = 8,
    ):
        sqrt2pi = np.sqrt(2 * np.pi)
        cellNum = graph.shape[0]
        clsNum = len(cls)

        with tqdm(range(max_iter), disable=not self.verbose) as pbar:
            for iter in pbar:
                # ICM step
                for _ in range(icm_iter):
                    temp_order = np.arange(cellNum)
                    changed = 0
                    np.random.shuffle(temp_order)
                    for i in temp_order:
                        newLabel = (labelList[i] + 1) % clsNum
                        temp_delta = self._delta_energy(
                            labelList, i, exp, graph, clsPara, newLabel, beta
                        )
                        if temp_delta < 0:
                            labelList[i] = newLabel
                            changed += 1
                    if changed == 0:
                        break

                # EM step initialize
                means, vars = clsPara.T
                vars[np.isclose(vars, 0)] = 1e-5
                expDiffSquared = (exp[:, None] - means) ** 2

                # E step Vectorized
                clusterProb = np.exp(-0.5 * expDiffSquared / vars) / (
                    sqrt2pi * np.sqrt(vars)
                )
                clusterProb = clusterProb / clusterProb.sum(axis=1)[:, None]

                # M Step Vectorized
                weights = clusterProb / clusterProb.sum(axis=0)
                means = np.sum(exp[:, None] * weights, axis=0)
                vars = np.sum(weights * expDiffSquared, axis=0) / weights.sum(axis=0)
                vars[np.isclose(vars, 0)] = 1e-5

                clsPara = np.column_stack([means, vars])

        return labelList

    def _delta_energy(self, labelList, index, exp, graph, clsPara, newLabel, beta):
        neighborIndices = graph[index].indices
        mean, var = clsPara[labelList[index]]
        newMean, newVar = clsPara[newLabel]
        sqrt_2_pi_var = np.sqrt(2 * np.pi * var)
        sqrt_2_pi_newVar = np.sqrt(2 * np.pi * newVar)

        delta_energy_const = (
            np.log(sqrt_2_pi_newVar / sqrt_2_pi_var)
            + ((exp[index] - newMean) ** 2 / (2 * newVar))
            - ((exp[index] - mean) ** 2 / (2 * var))
        )
        delta_energy_neighbors = beta * np.sum(
            self._difference(newLabel, labelList[neighborIndices])
            - self._difference(labelList[index], labelList[neighborIndices])
        )

        return delta_energy_const + delta_energy_neighbors

    @staticmethod
    def _difference(x, y):
        return np.abs(x - y)

from hmac import new
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy import  sparse as sp


class SingleGeneGraph:
    """
    Construct gene graph and implement HMRF in spatial transcriptomics
    """

    def __init__(
        self, gene_id: str,exp: np.ndarray, coord : np.ndarray,   kneighbors: int, verbose: bool = True,
    ):
        self.verbose = verbose
        self.exp = exp[:, gene_id] 
        self.cellNum = exp.shape[0]
        self.coord = coord 
        self.graph = self._construct_graph(self.coord, kneighbors)
        self.corr = self._get_corr(exp, n_comp=10)

    def mrf_with_icmem(self, beta, n_components=2, icm_iter=10, max_iter=10):
        """
        Implement HMRF with ICM-EM
        """
        gmm = GaussianMixture(n_components=n_components).fit(self.exp.reshape(-1, 1))
        means, covs = gmm.means_, gmm.covariances_
        pred = gmm.predict(self.exp).reshape(-1)
        cls = set(pred)
        clsPara = means.reshape(-1), covs.reshape(-1)
        clsPara = np.array(clsPara).T
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
        corrMatrix = corrMatrix / (corrMatrix.sum(axis=1).reshape(-1, 1) / alpha)
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
        nbrs = NearestNeighbors(n_neighbors=kneighbors).fit(coord)
        graph = nbrs.kneighbors_graph(coord)
        return graph

    @staticmethod
    def _get_corr(exp_matrix: np.ndarray, n_comp: int = 10):
        """
        Calculate the correlation between cells based on the principal components
        """
        scaler = StandardScaler()
        pca = PCA(n_components=n_comp)
        exp_matrix_scaled = scaler.fit_transform(exp_matrix)
        principal_components = pca.fit_transform(exp_matrix_scaled)
        return np.corrcoef(principal_components)

    def _label_resort(
        self, means, labelList
    ):  # Set the label with the highest mean as 1
        clsLabel = np.argmax(means[:, 0])
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
        icm_iter: int = 4,
        max_iter: int = 8,
    ):
        cellNum = graph.shape[0]
        clsNum = len(cls)
        changed = 0
        pbar = tqdm(range(max_iter)) if self.verbose else range(max_iter)
        for totalIter in pbar:
            if (changed == 0)  and (totalIter > 0):
                break
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
                    newLabel = np.abs(labelList[i] - 1) 
                    temp_delta = self._delta_energy(labelList, i, exp, graph, clsPara, newLabel, beta)
                    if temp_delta < 0:
                        labelList[i] = newLabel
                        delta += temp_delta
                        changed += 1
                iter += 1

            clusterProb = np.zeros([cellNum, clsNum])

            # E step Vectorized 
            means, vars = clsPara[:,0], clsPara[:,1]
            vars[np.isclose(vars, 0)] = 1e-5
            expDiffSquered = (exp[:, np.newaxis] - means)**2
            clusterProb = np.exp(-0.5 * expDiffSquered / vars) / np.sqrt(2 * np.pi * vars )
            clusterProb = clusterProb / np.sum(clusterProb, axis=1).reshape(-1, 1)
            # M Step Vectorized
            weights = clusterProb/ clusterProb.sum(axis=0, keepdims=True) 
            means = np.sum(exp[:, np.newaxis] * weights, axis=0)
            vars = np.sum(weights * expDiffSquered, axis=0) / weights.sum(axis=0)
            vars[np.isclose(vars, 0)] = 1e-5
            clsPara = np.stack([means, vars], axis=1)
        return labelList

    def _delta_energy(self, labelList, index, exp, graph, clsPara, newLabel, beta):
        neighborIndices = graph[index].indices
        mean, var = clsPara[labelList[index]]
        newMean, newVar = clsPara[newLabel]
        initEnergyConst = np.log(np.sqrt(2 * np.pi * var)) + (exp[index] -mean) ** 2 / (2* var)
        newEnergyConst = np.log(np.sqrt(2 * np.pi * newVar)) + (exp[index] - newMean) ** 2 / (2 * newVar)
        initEnergyNeighbors = beta * np.sum(self._difference(exp[index], exp[neighborIndices]))
        newEnergyNeighbors = beta * np.sum(self._difference(exp[index], exp[neighborIndices]))
        initEnergy = initEnergyConst + initEnergyNeighbors
        newEnergy = newEnergyConst + newEnergyNeighbors
        return newEnergy - initEnergy
         
    @staticmethod
    def _difference(x, y):
        return np.abs(x - y)



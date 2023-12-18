from asyncio import base_tasks
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from scipy import sparse as sp
import multiprocessing as mp


class SingleGeneGraph:
    """
    Construct gene graph and implement HMRF in spatial transcriptomics
    """

    def __init__(
        self,
        gene_id: str,
        exp: pd.DataFrame,
        graph: sp.csr_matrix = None,
        corr: np.ndarray = None,
        coord: np.ndarray = None,
        kneighbors: int = None,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.exp = exp.loc[:, gene_id].values
        self.cellNum = exp.shape[0]
        self.graph = (
            self._construct_graph(coord, kneighbors) if graph is None else graph
        )
        self.corr = self._get_corr(exp.values, n_comp=10) if corr is None else corr

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
        convengency_threshold: float = 1e-4,
    ):
        sqrt2pi = np.sqrt(2 * np.pi)
        cellNum = graph.shape[0]
        clsNum = len(cls)
        preClsPara = clsPara.copy()
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

                newClsPara = np.column_stack([means, vars])
                paraChange = np.max(np.abs(newClsPara - preClsPara))
                if paraChange < convengency_threshold:
                    if self.verbose:
                        print("Convergence reached")
                    break
                preClsPara = newClsPara

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


class MultiGeneGraph:
    def __init__(
        self,
        exp: pd.DataFrame,
        coord: np.ndarray,
        kneighbors: int,
        n_components: int = 2,
        beta: int = 3,
        alpha: float = 0.6,
        theta: float = 0.2,
        multiprocess: bool = True,
        NPROCESS: int = 3,
    ):
        self.exp = exp
        self.cellNum = exp.shape[0]
        self.geneList = exp.columns
        self.graph = self._construct_graph(coord, kneighbors)
        self.corr = self._get_corr(exp_matrix=exp.values, pca=10)
        self.n_components = n_components
        self.beta = beta
        self.alpha = alpha
        self.theta = theta
        self.pbar = tqdm(total=len(self.geneList))
        if multiprocess:
            self.NP = NPROCESS

    def get_impute(self, save=None):
        if self.NP == 1:
            for gene in self.geneList:
                graph = SingleGeneGraph(gene, self.exp, self.graph, self.corr)
                graph.mrf_with_icmem(self.beta, self.n_components)
                graph.impute(self.alpha, self.theta)
                self.pbar.update()
        else:
            manage = mp.Manager()
            imputedExpDict = manage.dict()
            labelDict = manage.dict()
            lock = manage.Lock()
            pool = mp.Pool(self.NP)
            for gene in self.geneList:
                pool.apply_async(
                    self._process_gene,
                    args=(
                        gene,
                        self.exp,
                        self.graph,
                        self.corr,
                        self.beta,
                        self.n_components,
                        self.alpha,
                        self.theta,
                        imputedExpDict,
                        labelDict,
                        lock,
                    ),
                    callback=self._update,
                )
            pool.close()
            pool.join()
        imputedExp = pd.DataFrame.from_dict(
            imputedExpDict, orient="index", columns=self.exp.index
        )
        imputedExp = imputedExp.T
        labelDict = pd.DataFrame.from_dict(
            labelDict, orient="index", columns=self.exp.index
        )
        labelDict = labelDict.T
        if save is not None:
            imputedExp.to_csv(save + "/imputedExp.csv")
            labelDict.to_csv(save + "/label.csv")
            return None
        else:
            return imputedExp, labelDict

    def _construct_graph(self, coord, kneighbors):
        return (
            NearestNeighbors(n_neighbors=kneighbors).fit(coord).kneighbors_graph(coord)
        )

    def _get_corr(self, exp_matrix, pca):
        return np.corrcoef(
            PCA(pca).fit_transform(StandardScaler().fit_transform(exp_matrix))
        )

    @staticmethod
    def _process_gene(
        gene,
        exp,
        graph,
        corr,
        beta,
        n_components,
        alpha,
        theta,
        imputedExpDict,
        labelDict,
        lock,
    ):
        graph = SingleGeneGraph(gene, exp, graph, corr)
        graph.mrf_with_icmem(beta, n_components)
        graph.impute(alpha, theta)
        with lock:
            imputedExpDict[gene] = graph.imputedExp.reshape(-1)
            labelDict[gene] = graph.label.reshape(-1)
        return gene

    def _update(self, args):
        self.pbar.update()

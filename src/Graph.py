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

    def mrf_with_icmem(
        self,
        beta,
        n_components=2,
        icm_iter=3,
        max_iter=10,
        update_exp=False,
        alpha=0.6,
        theta=0.2,
    ):
        """
        Implement HMRF with ICM-EM
        """
        self.n_components = n_components
        gmm = GaussianMixture(n_components=n_components).fit(self.exp.reshape(-1, 1))
        means, covs = gmm.means_.ravel(), gmm.covariances_.ravel()
        self.label = gmm.predict(self.exp.reshape(-1, 1))
        self.cls = range(n_components)
        self.clsPara = np.column_stack((means, covs))
        self._icmem(
            beta, icm_iter, max_iter, update_exp=update_exp, alpha=alpha, theta=theta
        )
        self._label_resort()
        if self.verbose:
            print(self.clsPara)

    def impute(self, alpha: float = 0.6, theta: float = 0.2):
        self.imputedExp = self._impute(alpha, theta)
        if self.verbose:
            print("Imputation finished")

    def _impute(self, alpha: float = 0.6, theta: float = 0.2):
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
        corrMatrix = abs(np.multiply(graph, self.corr))  # abs?
        neighborInform = corrMatrix - np.eye(corrMatrix.shape[0])
        neighborInform = alpha * (
            neighborInform / neighborInform.sum(axis=1).reshape(-1, 1)
        )
        adjacencyMatrix = neighborInform + np.eye(corrMatrix.shape[0])
        labelMatrix = (label.reshape(-1, 1) == label.reshape(1, -1)).astype(float)
        labelMatrix[labelMatrix == 0] = theta
        adjacencyMatrix = np.multiply(adjacencyMatrix, labelMatrix)
        adjacencyMatrix = adjacencyMatrix / adjacencyMatrix.sum(axis=1).reshape(-1, 1)
        imputedExp = np.matmul(adjacencyMatrix, self.exp)
        return imputedExp

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

    def _label_resort(self):
        # Set the label with the highest mean as 1
        means = self.clsPara[:, 0]
        clsLabel = np.argmax(means)
        newLabels = np.zeros_like(self.label)
        newLabels[self.label == clsLabel] = 1
        self.label = newLabels

    def _icmem(
        self,
        beta: float,
        icm_iter: int = 2,
        max_iter: int = 8,
        convengency_threshold: float = 1e-4,
        exp_update: bool = False,
        label_update: bool = False,
        alpha: float = 0.6,
        theta: float = 0.2,
    ):
        sqrt2pi = np.sqrt(2 * np.pi)
        cellNum = self.graph.shape[0]
        clsNum = len(self.cls)
        with tqdm(range(max_iter), disable=not self.verbose) as pbar:
            for iter in pbar:
                # ICM step
                for _ in range(icm_iter):
                    temp_order = np.arange(cellNum)
                    changed = 0
                    np.random.shuffle(temp_order)
                    for index in temp_order:
                        newLabel = (self.label[index] + 1) % clsNum
                        temp_delta = self._delta_energy(index, newLabel, beta)
                        if temp_delta < 0:
                            self.label[index] = newLabel
                            changed += 1
                    # print("changed: {} at iteratrion:{}".format(changed, iter))
                    if changed == 0:
                        break

                # EM step initialize
                means, vars = self.clsPara.T
                vars[np.isclose(vars, 0)] = 1e-5
                expDiffSquared = (self.exp[:, None] - means) ** 2

                # E step Vectorized
                clusterProb = np.exp(-0.5 * expDiffSquared / vars) / (
                    sqrt2pi * np.sqrt(vars)
                )
                clusterProb = clusterProb / clusterProb.sum(axis=1)[:, None]

                # M Step Vectorized
                weights = clusterProb / clusterProb.sum(axis=0)
                means = np.sum(self.exp[:, None] * weights, axis=0)
                vars = np.sum(weights * expDiffSquared, axis=0) / weights.sum(axis=0)
                vars[np.isclose(vars, 0)] = 1e-5

                newClsPara = np.column_stack([means, vars])
                paraChange = np.max(np.abs(newClsPara - self.clsPara))
                if paraChange < convengency_threshold:
                    if self.verbose:
                        print("Convergence reached at iteration {}".format(iter))
                    break
                self.clsPara = newClsPara

                # Update expression matrix
                # TODO : whether using GMM update labels or not
                if exp_update:
                    self.exp = self._impute(alpha=alpha, theta=theta)
                if label_update:
                    self.label = (
                        GaussianMixture(n_components=self.n_components)
                        .fit(self.exp.reshape(-1, 1))
                        .predict(self.exp.reshape(-1, 1))
                    )

        return

    def _delta_energy(self, index, newLabel, beta):
        neighborIndices = self.graph[index].indices
        mean, var = self.clsPara[self.label[index]]
        newMean, newVar = self.clsPara[newLabel]
        sqrt_2_pi_var = np.sqrt(2 * np.pi * var)
        sqrt_2_pi_newVar = np.sqrt(2 * np.pi * newVar)

        delta_energy_const = (
            np.log(sqrt_2_pi_newVar / sqrt_2_pi_var)
            + ((self.exp[index] - newMean) ** 2 / (2 * newVar))
            - ((self.exp[index] - mean) ** 2 / (2 * var))
        )
        delta_energy_neighbors = beta * np.sum(
            self._difference(newLabel, self.label[neighborIndices])
            - self._difference(self.label[index], self.label[neighborIndices])
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
        alpha: float = 0.1,
        theta: float = 0.2,
        multiprocess: bool = True,
        update: bool = False,
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
        self.update = update
        if multiprocess:
            self.NP = NPROCESS

    def get_impute(self, save=None):
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
                    self.update,
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
        update=False,
    ):
        graph = SingleGeneGraph(gene, exp, graph, corr)
        graph.mrf_with_icmem(
            beta,
            n_components,
            update_exp=update,
            alpha=alpha,
            theta=theta,
        )
        graph.impute(alpha, theta)
        with lock:
            imputedExpDict[gene] = graph.imputedExp.reshape(-1)
            labelDict[gene] = graph.label.reshape(-1)
        return gene

    def _update(self, args):
        self.pbar.update()

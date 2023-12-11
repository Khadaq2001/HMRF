import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.sparse as sp
from tqdm import tqdm

class SingleGeneGraph:
    """
    Construct gene graph and implement HMRF in spatial transcriptomics
    """

    def __init__(
        self, gene_id: str, exp: np.ndarray, coord: np.ndarray, kneighbors: int, verbose: bool = True,
    ):
        self.verbose = verbose
        self.exp = exp[:, gene_id]
        self.cellNum = exp.shape[0]
        self.coord = coord
        self.graph = self._construct_graph(self.coord, kneighbors)
        self.corr = self._get_corr(exp, n_comp=10)

    # No change in __init__
    ...

    def mrf_with_icmem(self, beta, n_components=2, icm_iter=10, max_iter=10):
        """
        Implement HMRF with ICM-EM
        """
        # Optimized the GMM initialization to avoid multiple reshape operations
        gmm = GaussianMixture(n_components=n_components).fit(self.exp.reshape(-1, 1))
        means, covs = gmm.means_.ravel(), gmm.covariances_.ravel()
        pred = gmm.predict(self.exp.reshape(-1, 1))
        clsPara = np.column_stack((means, covs))

        # Replaced the set 'cls' with a pre-calculated range based on 'n_components'
        range_n_components = np.arange(n_components)
        labelList = self._icmem(pred, beta, range_n_components, clsPara, self.exp, self.graph, icm_iter, max_iter)
        if self.verbose:
            print(clsPara)
        labelList = self._label_resort(means, labelList)
        self.label = labelList

    # No change in impute function
    ...

    @staticmethod
    def _construct_graph(coord: np.ndarray, kneighbors: int = 6):
        """
        Construct gene graph based on the nearest neighbors
        """
        # Optimized graph construction by not calling fit method separately
        graph = NearestNeighbors(n_neighbors=kneighbors).fit(coord).kneighbors_graph(coord)
        return graph

    @staticmethod
    def _get_corr(exp_matrix: np.ndarray, n_comp: int = 10):
        """
        Calculate the correlation between cells based on the principal components
        """
        # Combined the steps for scaling and PCA transformation
        return np.corrcoef(PCA(n_comp).fit_transform(StandardScaler().fit_transform(exp_matrix)))

    # No change in _label_resort function
    ...

    def _icmem(
        self,
        labelList: np.ndarray,
        beta: float,
        cls: np.ndarray,
        clsPara: np.ndarray,
        exp: np.ndarray,
        graph: sp.csr_matrix,
        icm_iter: int = 4,
        max_iter: int = 8,
    ):
        # Pre-calculating constants for Gaussian distributions outside the loop for efficiency
        sqrt_2_pi = np.sqrt(2 * np.pi)
        cellNum = graph.shape[0]
        clsNum = cls.size
        changed = 0

        # Using with statement with tqdm to enhance readability and ensure proper closure
        with tqdm(range(max_iter), disable=not self.verbose) as pbar:
            for totalIter in pbar:
                if (changed == 0) and (totalIter > 0):
                    break

                # ICM step
                temp_order = np.arange(cellNum)
                np.random.shuffle(temp_order)
                for i in temp_order:
                    newLabel = (labelList[i] + 1) % clsNum  # Assuming binary classes, changed from abs(x - 1) for sustainability in multi-class scenario
                    temp_delta = self._delta_energy(labelList, i, exp, graph, clsPara, newLabel, beta)
                    if temp_delta < 0:
                        labelList[i] = newLabel
                        changed += 1

                # Pre-computing expressions for the Gaussian likelihood calculation
                means, vars = clsPara.T  # Avoid reshaping in each iteration
                vars = np.clip(vars, 1e-5, np.inf)  # Ensure no division by zero
                expDiffSquared = (exp[:, None] - means) ** 2

                # E step Vectorized 
                clusterProb = np.exp(-0.5 * expDiffSquared / vars) / (sqrt_2_pi * np.sqrt(vars))
                clusterProb /= clusterProb.sum(axis=1)[:, None]
                
                # M Step Vectorized
                weights = clusterProb / clusterProb.sum(axis=0)
                means = (exp[:, None] * weights).sum(axis=0)
                vars = (weights * expDiffSquared).sum(axis=0) / weights.sum(axis=0)
                vars = np.clip(vars, 1e-5, np.inf)  # Ensure variances are not too small

                clsPara = np.column_stack((means, vars))

                if not changed:
                    break
                
        return labelList

    def _delta_energy(self, labelList, index, exp, graph, clsPara, newLabel, beta):
        # Computation of energies is vectorized and simplified to increase efficiency
        neighborIndices = graph[index].indices
        mean, var = clsPara[labelList[index]]
        newMean, newVar = clsPara[newLabel]
        sqrt_2_pi_var = np.sqrt(2 * np.pi * var)
        sqrt_2_pi_newVar = np.sqrt(2 * np.pi * newVar)
        
        # Calculate only the change in energy, avoiding full energy calculation
        delta_energy_const = np.log(sqrt_2_pi_newVar / sqrt_2_pi_var) + ((exp[index] - newMean) ** 2 / newVar) - ((exp[index] - mean) ** 2 / var) / 2
        delta_energy_neighbors = beta * (np.sign(labelList[neighborIndices] - labelList[index]) != np.sign(labelList[neighborIndices] - newLabel)).sum()
        
        return delta_energy_const + delta_energy_neighbors

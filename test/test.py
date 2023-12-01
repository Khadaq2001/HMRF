import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def readSpatialExpression(
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
    coord = pd.DataFrame(coord, columns=["x", "y"])
    coord.index = data.index
    return coord, data, noiseInd


def getCorr(principalComponents):
    corr = np.corrcoef(principalComponents)
    return corr


def main():
    n_component = 10
    dataDir = "../../dataset/MOB-breast_cancer/Rep11_MOB_count_matrix-1.tsv"
    locs, data, _ = readSpatialExpression(dataDir)
    print(data.shape)

    scaler = StandardScaler()
    pca = PCA(n_components=n_component)
    dataScaled = scaler.fit_transform(data)
    principalComponents = pca.fit_transform(dataScaled)
    principalComponents = pd.DataFrame(principalComponents)
    print(principalComponents)
    corr = getCorr(principalComponents)
    print(corr)
    

if __name__ == "__main__":
    main()

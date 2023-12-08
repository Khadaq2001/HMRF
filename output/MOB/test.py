import numpy as np
import pandas as pd
import scanpy as sc
from src.Graph import SingleGeneGraph
from src.Function import read_spatial_expression
from src import Preprocess
from tqdm import tqdm


def main():
    kneighbors, beta = 4, 2
    dataDir = "../dataset/MOB-breast_cancer/Rep11_MOB_count_matrix-1.tsv"
    locs, data, _ = read_spatial_expression(dataDir)
    locsDf = pd.DataFrame(locs, columns=["imagerow", "imagecol"])
    locsDf.index = data.index
    adata = sc.AnnData(X=data, obs=locsDf)
    adata = Preprocess.data_preprocess(adata, high_var=True)
    new_index = ["array_row", "array_col"]
    adata.obs.rename(
        columns={"imagerow": new_index[0], "imagecol": new_index[1]}, inplace=True
    )
    imputedExp = pd.DataFrame(index=adata.obs.index, columns=adata.var_names)
    pbar = tqdm(adata.var_names)
    for gene in pbar:
        graph = SingleGeneGraph(adata, gene, kneighbors, verbose=False)
        graph.mrf_with_icmem(beta=beta, icm_iter=3, max_iter=5)
        graph.impute(alpha=0.6, theta=0.2)
        imputedExp[gene] = graph.imputedExp
    imputedExp.to_csv("imputedExp.csv")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import os
import scanpy as sc
from src.Graph import MultiGeneGraph
from datetime import date

today = date.today().strftime("%m%d")


def main():
    adata = sc.read_h5ad("dataset/DLPFC/processed.h5ad")
    print(adata.shape)
    exp = adata.to_df()
    coord = adata.obsm["spatial"]
    path = f"./output/DLPFC/{today}"
    os.makedirs(path, exist_ok=True)
    geneGraph = MultiGeneGraph(
        exp=exp,
        coord=coord,
        kneighbors=6,
        NPROCESS=3,
        alpha=0.8,
        theta=0.2,
        max_iter=10,
        exp_update=False,
        label_update=False,
    )
    geneGraph.get_impute(save=path)


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import os
import scanpy as sc
from src import Preprocess
from src.Graph import MultiGeneGraph
from datetime import date

today = date.today().strftime("%m%d")


def main():
    adata = sc.read_h5ad("dataset/DLPFC/processed.h5ad")
    exp = adata.to_df()
    coord = adata.obsm["spatial"]
    path = f"./output/{today}WithImputed"
    os.makedirs(path, exist_ok=True)
    geneGraph = MultiGeneGraph(
        exp=exp,
        coord=coord,
        kneighbors=18,
        NPROCESS=3,
        alpha=0.6,
        theta=0.2,
        update=True,
    )
    geneGraph.get_impute(save=path)


if __name__ == "__main__":
    main()

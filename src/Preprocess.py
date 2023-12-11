import scanpy as sc
import numpy as np
import pandas as pd
import os


def read_data(path):
    """
    Read data from the project folder
    """
    adata = sc.read_visium(path)
    adata.var_names_make_unique()
    rows, cols = adata.shape
    print(f"Read data with {rows} cells, {cols} genes")
    return adata


def data_preprocess(adata, min_genes=200, min_cells=3, high_var=False, n_top=None):
    """preprocessing adata"""
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if high_var:
        if n_top:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top)
        else:
            sc.pp.highly_variable_genes(
                adata, min_mean=0.0125, max_mean=3, min_disp=0.5
            )
        adata.raw = adata
        adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)

    return adata


def extract_data(adata):
    return pd.DataFrame(adata.X), pd.DataFrame(adata.obsm["spatial"])


def main():
    dataset = "/home/qinxianhan/project/spatial/dataset"
    data_path = os.path.join(dataset, "Mouse_brain/Data")
    adata = read_data(data_path)
    # print(adata.shape)


if __name__ == "__main__":
    main()

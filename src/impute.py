import numpy as np
import pandas as pd
import scanpy as sc


def impute(
    exp: np.ndarray,
    label_list: np.ndarray,
    threshld: float,
    shift: float,
):
    """
    Impute single gene expression spatial transcriptomics data from same label cells.
    """
    for label in label_list:
        idx = np.where(label_list == label)
        cluster_exp = exp[:, idx]
        min_index = np.argmin(cluster_exp)
        cluster_exp[min_index] = np.mean(cluster_exp)
        exp[:, idx] = cluster_exp + shift
    return exp


def main():
    test_data = pd.read_csv("./temp/test_data.csv")
    test_data = test_data.values
    test_label = pd.read_csv("./test_label.csv")
    print(test_data)
    imputed_data = impute(test_data, test_label, 0.1, 0.1)
    print(imputed_data)


if __name__ == "__main__":
    main()

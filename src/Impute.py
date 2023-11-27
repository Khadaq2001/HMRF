import numpy as np
import pandas as pd



def impute(
    exp: np.ndarray,
    label_list: np.ndarray,
    threshld: float,
    shift: float,
):
    """
    Impute single gene expression spatial transcriptomics data from same label cells.
    """
    for label in set(label_list):
        idx = np.where(label_list == label)
        cluster_exp = exp[idx]
        min = np.min(cluster_exp)
        min_index = np.where(cluster_exp == min)
        print(min_index)
        print(f"imputed spot: {len(min_index[0])}")
        cluster_exp[min_index] = np.mean(cluster_exp)
        exp[idx] = cluster_exp + shift
    return exp


def main():
    test_data = pd.read_csv("../temp/test_data.csv").values.reshape(-1)
    test_label = pd.read_csv("../temp/test_labels.csv").values.reshape(-1)
    print(test_label)
    imputed_data = impute(test_data, test_label, 0.1, 0.1)
    print(imputed_data)


if __name__ == "__main__":
    main()

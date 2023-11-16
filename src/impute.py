import numpy as np
import pandas as pd
import scanpy as sc

def impute(
    exp: np.ndarray,
    label_list : np.ndarray,
    threshld: float,
):
    """
    Impute single gene expression spatial transcriptomics data from same label cells.
    """
    for label in label_list:
        idx = np.where(label_list == label)
        cluster_exp = exp[:,idx ]
        
def main():
       
    ...
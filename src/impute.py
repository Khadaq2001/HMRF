import numpy as np
import pandas as pd
import scanpy as sc

def impute(
    cluster_exp: ndarrary,
    threshld: float,

)
    """
    Impute single gene expression spatial transcriptomics data from same label cells.
    """
    cluster_exp[cluster_exp < threshld] = np.mean(cluster_exp)
    return cluster_exp


def main():
    

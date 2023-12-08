from gc import callbacks
from unittest import result
import numpy as np
import pandas as pd
import scanpy as sc
from src import Preprocess
from src.Graph import SingleGeneGraph
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

kneighbors, beta = 6, 3
NCPU = 28


def process_gene(gene, adata):
    graph = SingleGeneGraph(adata, gene, kneighbors, verbose=False)
    graph.mrf_with_icmem(beta=beta, icm_iter=3, max_iter=5)
    graph.impute(alpha=0.6, theta=0.2)
    return graph.imputedExp

    
def main():
    outDir = "./output/goldStandard"
    adata = sc.read_visium("../dataset/goldStandard")
    adata.var_names_make_unique()
    adata = Preprocess.data_preprocess(adata, high_var=True)
    geneList = adata.var_names
    with mp.Pool(NCPU) as pool:
        result = pool.map(partial(process_gene, adata=adata), geneList )
        pool.close()
        pool.join()
    for gene, exp in zip(geneList, result):
        imputedExp[gene] = exp
    imputedExp = pd.DataFrame(index=adata.obs.index, columns=adata.var_names)
    imputedExp.to_csv(f"{outDir}/goldStandard")


if __name__ == "__main__":
    main()

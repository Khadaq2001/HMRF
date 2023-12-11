import numpy as np
import pandas as pd
import scanpy as sc
from src import Preprocess
from src.Graph import SingleGeneGraph
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Manager

kneighbors, beta = 6, 3
NP = 3
adata = sc.read_visium("../dataset/goldStandard")
adata.var_names_make_unique()
adata = Preprocess.data_preprocess(adata, high_var=True, n_top=3000)
geneList = adata.var_names
pbar = tqdm(total=len(geneList))


def process_gene(gene, imputedExpDict, lock):
    graph = SingleGeneGraph(adata, gene, kneighbors, verbose=False)
    graph.mrf_with_icmem(beta=beta, icm_iter=3, max_iter=5)
    graph.impute(alpha=0.6, theta=0.2)
    with lock:
        imputedExpDict[gene] = graph.imputedExp.reshape(-1)
    return gene


def update(args):
    # gene, imputedExpDict = args
    pbar.update()


def main():
    manage = Manager()
    imputedExpDict = manage.dict()
    lock = manage.Lock()

    pool = mp.Pool(NP)
    for gene in geneList:
        pool.apply_async(
            process_gene, args=(gene, imputedExpDict, lock), callback=update
        )

    pool.close()
    pool.join()
    imputedExp = pd.DataFrame.from_dict(
        imputedExpDict, orient="index", columns=adata.obs.index
    )
    imputedExp = imputedExp.T
    imputedExp.to_csv("imputedExp.csv")

if __name__ =="__main__":
    main()

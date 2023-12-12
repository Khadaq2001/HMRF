from cProfile import label
import numpy as np
import pandas as pd
import scanpy as sc
from src import Preprocess
from src.Graph import SingleGeneGraph
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Manager


def preprocess_gene(n_top=3000):
    adata = sc.read_visium("../dataset/goldStandard")
    adata.var_names_make_unique()
    adata = Preprocess.data_preprocess(adata, high_var=True, n_top=n_top)
    exp = adata.to_df()
    coord = adata.obsm["spatial"]
    exp.to_csv("input/gsExp.csv")
    np.savetxt("input/gsCoord.csv", coord, delimiter=",", fmt="%d")


# preprocess_gene()
kneighbors, beta = 18, 3
NP = 3
exp = pd.read_csv("input/gsExp.csv", index_col=0)
coord = np.genfromtxt("input/gsCoord.csv", delimiter=",", dtype=int)
geneList = exp.columns
pbar = tqdm(total=len(geneList))


def process_gene(gene, imputedExpDict, labelDict, lock):
    graph = SingleGeneGraph(gene, exp, coord, kneighbors, verbose=False)
    graph.mrf_with_icmem(beta=beta, icm_iter=3, max_iter=8)
    graph.impute(alpha=0.8, theta=0.2)
    with lock:
        imputedExpDict[gene] = graph.imputedExp.reshape(-1)
        labelDict[gene] = graph.label.reshape(-1)
    return gene


def update(args):
    # gene, imputedExpDict = args
    pbar.update()


def main():
    manage = Manager()
    imputedExpDict = manage.dict()
    labelDict = manage.dict()
    lock = manage.Lock()
    pool = mp.Pool(NP)
  #  print(exp)
  #  print(coord)
    for gene in geneList:
        # print(gene)
        pool.apply_async(
            process_gene, args=(gene, imputedExpDict, labelDict, lock), callback=update
        )

    pool.close()
    pool.join()

    imputedExp = pd.DataFrame.from_dict(
        imputedExpDict, orient="index", columns=exp.index
    )
    imputedExp = imputedExp.T
    imputedExp.to_csv("./output/imputedExp.csv")
    labelDict = pd.DataFrame.from_dict(labelDict, orient="index", columns=exp.index)
    labelDict = labelDict.T
    labelDict.to_csv("./output/label.csv")


if __name__ == "__main__":
    main()

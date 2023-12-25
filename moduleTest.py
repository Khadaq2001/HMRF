import numpy as np
import pandas as pd
import os
from src import Preprocess
from src.Graph import MultiGeneGraph
from datetime import date

today = date.today().strftime("%m%d")


def main():
    exp = pd.read_csv("dataset/DLPFC/expFull.csv", index_col=0)
    coord = np.genfromtxt("dataset/DLPFC/gsCoord.csv", delimiter=",", dtype=int)
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

import numpy as np
import pandas as pd
from src import Preprocess
from src.Graph import MultiGeneGraph


def main():
    exp = pd.read_csv("dataset/gsExp.csv", index_col=0)
    coord = np.genfromtxt("dataset/gsCoord.csv", delimiter=",", dtype=int)
    geneGraph = MultiGeneGraph(exp=exp, coord=coord, kneighbors=6, NPROCESS=3)
    geneGraph.get_impute()


if __name__ == "__main__":
    main()

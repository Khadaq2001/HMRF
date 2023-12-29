# Markov Random Field for ST Data 

## Discription 

This project introduce a statistic model using Hidden Markov Raodom Field (HMRF) for analyzing Spatial Transcriptomics Data. 
The model procecss and cluster every single variable (gene/feature) to N component (default 2) considering neighbor information.

## Usage
### Input data 

Model mainly takes expression matrix and spatial coordinates array as input. You can get them from your anndata:

```python
import scanpy as sc
adata = sc.read_h5ad(h5adPath) 
exp = adata.to_df()
coord = adata.obsm['spatial']
targetGene = "CAMK2N1" #For example 
```

### Graph for single gene 
Class `SingleGeneGraph` create a single_gene_graph object, which takes gene expression matrix, spatial coordinates (2-d array) and target gene(feature) name as input, and meanwhile generate a graph based on the spot's physical distance if no graph imformation is given. Afterwards, use class method 'mrf_with_iceme()' and 'impute()' to process and generate the output label and imputed expression.

```python
from src.Graph import SingleGeneGraph 
beta, kneighbors = 3, 6
graph = SingleGeneGraph(exp, targetGene, coord, kneighbors)
graph.mrf_with_icmem(beta)
graph.impute()
outputLabel, imputedExp = graph.label, graph.imputedExp
```

### Graph for all gene
Class `MultiGeneGraph` creates a muilti_gene_graph object, which takes expression matrix and spatial coordinates as input. Generate and process gene sequencially. It's recommanded to run in multi process way because it's unacceptable slow at present stage :( . 

```python
from src.Graph import MultiGeneGraph
graph= MultiGeneGraph(
        exp=exp,
        coord=coord,
        kneighbors=6,
        NPROCESS=3,
        alpha=0.8,
        theta=0.2,
        max_iter=10,
        exp_update=True,
        label_update=False,
    )
    geneGraph.get_impute(save=path)
```

More detailed tutorial is under progress.
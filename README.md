# Markov Random Field for ST Data 

## Discription 

A statistic model using Hidden Markov Raodom Field (HMRF) for analyzing Spatial Transcriptomics Data. 
The model procecss and cluster every single variable (gene/feature) to N component (default 2) considering neighbor information.

Class `SingleGeneGraph` create single graph object, which takes single gene expression and spatial coordinates as input and meanwhile generate a graph based on the spot's physical distance. 
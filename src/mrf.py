import numpy as np
import pandas as pd 
import scanpy as sc
import parmap #type: ignore 
import multiprocessing as mp
import random
from sklearn import mixture 


def difference(x, y):
    return np.abs(x-y)  

def init_energy(labels_mtx, pixels, beta, cls_para, neighbor_indices):
    w = labels_mtx 
    energy = 0.
    rows, cols = w.shape 
    for i in range(rows):
        for j in range(cols):
            mean, var= cls_para[w[i,j]]
            energy += np.log(np.sqrt(2*np.pi*var))
            energy += (pixels[i,j] - mean)**2/(2*var)
            for a, b in neighbor_indices:
                a +=i
                b +=j
                if 0 <=a < rows and 0 <= b < cols:
                    energy += beta*difference(w[i,j], w[a,b])
    return energy  

def delta_energy(labels_mtx,pixels, index,new_label, beta, cls_para, neighbor_indices):
    w = labels_mtx
    (i, j) = index 
    rows, cols = w.shape
    mean, var = cls_para[w[i,j]]
    init_energy = np.log(np.sqrt(2*np.pi*var)) + (pixels[i,j] - mean) ** 2/(2*var)
    for a, b in neighbor_indices:
        a += i
        b += j
        if 0 <=a < rows and 0 <= b < cols:
            init_energy += beta*difference(w[i,j], w[a,b])      
    mean_new , var_new = cls_para[new_label]
    new_energy =  np.log(np.sqrt(2*np.pi*var_new)) + (pixels[i,j] - mean_new) ** 2/(2*var_new)
    for a, b in neighbor_indices:
        a += i
        b += j
        if 0 <=a < rows and 0 <= b < cols:
            new_energy += beta*difference(new_label, w[a,b])
    # print (new_energy, init_energy)
    return new_energy - init_energy

def annealing(labels_mtx, cls, temp_function,
              pixels, beta, cls_para, neighbor_indices, max_iteration=10000,
              initial_temp = 1000):
    w =  labels_mtx
    (rows, cols) = w.shape
    current_energy = init_energy(w, pixels, beta, cls_para ,neighbor_indices)
    current_tmp = initial_temp
    changed = 0
    iter = 0
    while(iter<max_iteration):
        i = random.randint(0,rows-1)
        j = random.randint(0,cols-1)
        cls_list = list(cls)
        r = random.randint(0,len(cls_list) - 1)
        new_value = cls_list[r]
        delta = delta_energy(w, pixels, (i,j), new_value, beta, cls_para, neighbor_indices)

        r = random.uniform(0,1)
        
        if(delta < 0):
            w[i, j] = new_value
            current_energy += delta
            changed +=1
        else:
            try:
                if (-delta / current_tmp<-600):
                    k=0
                else :
                    k = np.exp(-delta / current_tmp)
            except:
                k = 0

            if r < k :
                w [i, j] = new_value
                current_energy += delta 
        if (temp_function):
            current_tmp = temp_function(current_tmp) 
        iter +=1
    print(f"{changed} pixels changed after {iter} iterations")  
    return w
        

def mrf_process(adata, gene_id,beta, n_components = 2, temp_function = lambda x: 0.99*x,
        max_iteration=10000, neighbor_indice =[(-1,1),(1,1),(1,-1),(1,1)]):
    """
    Marfov random field complete part
    """
    coord = adata.obs[['array_row','array_col']].values
    exp = adata[:,gene_id].X.toarray()
    rows, cols = np.max(coord, axis=0) 
    pixels = np.zeros((rows, cols))
    labels_mtx = np.zeros((rows, cols), dtype=int)
    label_list = np.zeros(len(coord)) 
    gmm = mixture.GaussianMixture(n_components=n_components)
    gmm.fit(exp)
    labels = gmm.predict(exp)
    for i,(x,y) in enumerate(coord):
        pixels[x-1,y-1] = exp[i]
        labels_mtx[x-1,y-1] = labels[i]
    cls_para = gmm.means_.reshape(-1), gmm.covariances_.reshape(-1)
    cls_para = np.array(cls_para).T
    labels_mtx = annealing(labels_mtx, labels, temp_function, pixels, beta, cls_para, neighbor_indice, max_iteration) 
    for i, (x,y) in enumerate(coord):
       label_list[i] = labels_mtx[x-1,y-1] 
    return label_list
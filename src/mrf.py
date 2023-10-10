import numpy as np
import pandas as pd 
import scanpy as sc
import parmap #type: ignore 
import multiprocessing as mp
import random
from sklearn import mixture 


def difference(x, y):
    return np.abs(x-y)  

def init_energy(labels, pixels, beta, cls_info, neighbor_indices):
    w = labels 
    energy = 0.
    rows, cols = w.shape 
    for i in rows:
        for j in cols:
            mean = cls_info[w[i,j]][0]
            var = cls_info[w[i,j]][1]
            energy += np.log(np.sqrt(2*np.pi*var))
            energy += (pixels[i,j] - mean)**2/(2*var)
            for a, b in neighbor_indices:
                a +=i
                b +=j
                if 0 <=a < rows and 0 <= b < cols:
                    energy += beta*difference(w[i,j], w[a,b])
    return energy  

def delta_energy(labels,pixels, index,new_label, beta, cls_info, neighbor_indices):
    w = labels
    (i, j) = index 
    rows, cols = w.shape
    mean, var = cls_info[w[i,j]]
    init_energy = np.log(np.sqrt(2*np.pi*var)) + (pixels[i,j] - mean) ** 2/(2*var)
    for a, b in neighbor_indices:
        a += i
        b += j
        if 0 <=a < rows and 0 <= b < cols:
            init_energy += beta*difference(w[i,j], w[a,b])      
    mean_new , var_new = cls_info[new_label]
    new_energy =  np.log(np.sqrt(2*np.pi*var_new)) + (pixels[i,j] - mean_new) ** 2/(2*var_new)
    for a, b in neighbor_indices:
        a += i
        b += j
        if 0 <=a < rows and 0 <= b < cols:
            new_energy += beta*difference(new_label, w[a,b])

    return new_energy - init_energy

def label_update (init_w, labels, temp_function,
              pixels, beta, cls_info, neighbor_indices, max_iteration=10000,
              initial_temp = 1000):
    w = np.array(init_w)
    rows, cols = w.shape
    current_energy = init_energy(w, pixels, beta, cls_info ,neighbor_indices)
    current_tmp = initial_temp
    iter = 0
    while(iter<max_iteration):
        i = random.randint(0,rows-1)
        j = random.randint(0,cols-1)
        label_list = list(labels)
        r = random.randint(0,len(label_list) - 1)
        new_value = label_list[r]
        delta = delta_energy(labels, pixels, (i,j), new_value, beta, cls_info, neighbor_indices)

        r = random.uniform(0,1)
        
        if(delta < 0):
            w[i, j] = new_value
            current_energy += delta
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
            
        iter +=1
        
        return w
        

    

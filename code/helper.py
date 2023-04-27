import os
import numpy as np
import pandas as pd
import CVIs
import methods
import inspect
from itertools import combinations

''' LOADING THE DATA '''
def load_data(file_path:str) -> np.array:
    D = pd.read_csv(file_path, delim_whitespace=True).to_numpy()
    data, labels = D[:,:-1], D[:,-1]
    return data,labels


''' CONSTRAINT GENERATION METHODS '''
def percentage_constrint_generation(data:np.array, GT_labels:np.array, percentage:float) -> np.array:
    ML, CL = [], []
    N,n = data.shape
    combs = list(combinations(np.arange(N), 2))
    no_combs = len(list(combs))
    no_constraints = int(percentage*N)
    rand_ind = np.random.permutation(np.arange(no_combs))[:no_constraints]
    combs = np.asarray(combs)

    for comb in combs[rand_ind, :]:
        if GT_labels(comb[0]) == GT_labels(comb[1]):
            ML.append(comb)
        else:
            CL.append(comb)
    return ML, CL

''' FUNCTIONS TO GATHER THE REQUIRED ELEMENTS OF THE EXPERIMENT '''
def get_CVIs() -> list:
    return [f for _, f in inspect.getmembers(CVIs, inspect.isfunction)]

def get_clMethods() -> list:
    return [f for _, f in inspect.getmembers(methods, inspect.isfunction)]

def get_datasets() -> list:
    ds_loc = '../datasets/'
    datasets = []
    for file in os.listdir(ds_loc):
        if file.endswith('.txt'):
            datasets.append(ds_loc + file)
    return datasets
import os
import numpy as np
import pandas as pd
import CVIs
import methods
import inspect
from itertools import combinations
import random


''' LOADING THE DATA '''
def load_data(file_path:str) -> np.array:
    D = pd.read_csv(file_path).to_numpy()
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

    return np.asarray(ML), np.asarray(CL)

def random_constraint_generation(data:np.array, GT_lables:np.array, percentage:float) -> np.array:
    ML,CL = [], []
    N, n = data.shape
    no_constraints = int(N*percentage)
    count = 0
    while count < no_constraints:
        p1, p2 = 0, 0
        while p1 == p2:
            p1, p2 = random.randint(0,N-1), random.randint(0,N-1)
        if GT_lables[p1] == GT_lables[p2]:
            ML.append([p1, p2])
        else:
            CL.append([p1, p2])
        count += 1
    return np.asarray(ML), np.asarray(CL)


def constraint_generation_setN(data:np.array, GT_labels:np.array, N:int) -> np.array:
    ML, CL = [], []
    clusters = {}
    unique_labels = np.unique(GT_labels)
    for i in unique_labels:
        this_cluster = (GT_labels == i)
        clusters[i] = np.asarray(this_cluster).nonzero()[0]
    nc = 0
    while nc < N:
        #  generate a ML constraint
        rc = random.sample(set(unique_labels),1)

        ml_pair = random.sample(set(clusters[rc[0]]), 2)
        ML.append(ml_pair)

        rp = random.sample(set(unique_labels), 2)
        cl_pair = []
        for i in np.arange(2):
            cl_pair.append(random.sample(set(clusters[rp[i]]), 1)[0])
        CL.append(cl_pair)
        nc += 2
    return np.array(ML),np.array(CL)

''' FUNCTIONS TO GATHER THE REQUIRED ELEMENTS OF THE EXPERIMENT '''
def get_CVIs() -> list:
    return [f for _, f in inspect.getmembers(CVIs, inspect.isfunction)]

def get_clMethods() -> list:
    return [f for _, f in inspect.getmembers(methods, inspect.isfunction)]

def get_datasets(ds_loc:str) -> list:
    datasets = []
    for file in os.listdir(ds_loc):
        if file.endswith('.csv'):
            datasets.append(ds_loc + file)
    return datasets
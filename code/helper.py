import os
import numpy as np
import pandas as pd
import CVIs
import methods
import inspect

''' LOADING THE DATA '''
def load_data(file_path:str) -> np.array:
    D = pd.read_csv(file_path, delim_whitespace=True).to_numpy()
    data, labels = D[:,:-1], D[:,-1]
    return data,labels


''' CONSTRAINT GENERATION METHODS '''
def RAL(data:np.array, GT_labels:np.array) -> np.array:
    return

def RAC(data:np.array, GT_labels:np.array) -> np.array:
    return


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
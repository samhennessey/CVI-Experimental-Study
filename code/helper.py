import os
import numpy as np
import pandas as pd
import CVIs
import methods

def load_data(file_path:str) -> np.array:
    D = pd.read_csv(file_path, delim_whitespace=True).to_numpy()
    data, labels = D[:,:-1], D[:,-1]
    return data,labels

def get_CVIs() -> list:
    return [f for _,f in CVIs.__dict__.iteritems() if callable(f)]

def get_clMethods() -> list:
    return [f for _,f in methods.__dict__.iteritems() if callable(f)]

def get_datasets() -> list:
    ds_loc = '../datasets/'
    datasets = []
    for file in os.listdir(ds_loc):
        if file.endswith('.txt'):
            datasets.append(ds_loc + file)
    return datasets
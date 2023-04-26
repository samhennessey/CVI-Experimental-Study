''' 
Author: Sam Hennessey
Date: 26/4/2023
'''

from helper import *

if __name__ == '__main__':

    # get a list of all datasets....
    datasets = get_datasets()
    
    # get a list of all algorithms.....
    clMethods = get_clMethods()

    # get a list of all CVIs
    cvis = get_CVIs()

    for ds in datasets: # for every dataset
        D, L = load_data(ds)
        for method in clMethods:
            vars = method.__code__.co_varnames
            

            



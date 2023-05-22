import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import re
import json

def display_results(results_path:str, folder_re:str, ALG:list, RUNS:np.array, cvis:list, CI:float):
    results = gather_results(results_path, folder_re, ALG, RUNS, cvis)
    for i in np.arange(ALG):
        NMI_values = results[i, :, :]
        table = generate_table(NMI_values, cvis, CI)
        table.to_csv(results_path + '/results.csv')
        plot_results(cvis, np.mean(NMI_values, axis=0), ALG)

def plot_results(CVI_names:list, avg_NMI:np.array, ALG_names:list):
    plt.figure(figsize=(20,5))
    plt.title('Average NMI of chosen partition according to CVIs')
    plt.ylabel('AVG NMI')
    plt.xlabel('CVI')
    plt.ylim([0,1])
    for i in np.arange(len(ALG_names)):
        plt.plot(CVI_names, np.mean(avg_NMI[i,:], axis=0),'.-', label=ALG_names[i])
    plt.grid('on')
    plt.legend()
    plt.show()

def gather_results(results_path:str, folder_re:str, ALG:list, RUNS:np.array, cvis:list):

    results = np.empty((len(ALG), len(RUNS), len(cvis)))

    run_ind = 0
    for folder in sorted(os.listdir(results_path)): # list all folders inside the results folder

        pattern = re.compile(folder_re)
        if pattern.match(folder): # if the folder matches the pattern 'NP_...'
            ALGORITHMS = results_path + '/' + folder

            alg_ind = 0
            for alg in sorted(os.listdir(ALGORITHMS)):

                alg_results = []

                CVIS = ALGORITHMS + '/' + alg
                for cvi in sorted(os.listdir(CVIS)):

                    cvi_pattern = re.compile('.*?.csv$')
                    if not cvi_pattern.match(cvi):

                        CVI_FOLDER = CVIS + '/' + cvi
                        for data_loc in sorted(os.listdir(CVI_FOLDER)):
                            data_pattern = re.compile('.*?.txt$')
                            if data_pattern.match(data_loc):

                                with open(CVI_FOLDER + '/' + data_loc) as infile:
                                    data = json.load(infile)
                                    alg_results.append(data['NMI'])
                
                results[alg_ind, run_ind, :] = alg_results
                alg_ind += 1

            run_ind += 1

    return results

def generate_table(CVI_values:np.array, CVI_legend:list, CI:float) -> pd.DataFrame:

    df = pd.DataFrame()
    for i in np.arange(CVI_values.shape[1]):

        # Calculate the confidence interval
        CI_95 = st.norm.interval(alpha=CI, loc=np.mean(CVI_values[:,i]), scale=st.sem(CVI_values[:,i]))
        rd = {'Mean_CVI': np.mean(CVI_values[:,i]), 'Standard_Deviation':np.std(CVI_values[:,i]), 'CI_lower':CI_95[0], 'CI_higher':CI_95[1]}  
        curr_df = pd.DataFrame(data=rd, index=[CVI_legend[i]])
        df = pd.concat([df, curr_df], ignore_index=False)    

    return df
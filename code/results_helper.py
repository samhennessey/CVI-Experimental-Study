import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import re
import json

def display_results(results_path:str, folder_re:str, ALG:list, RUNS:np.array, cvis:list, CI:float):
    ALG, cvis = sorted(ALG), sorted(cvis)
    results = gather_results(results_path, folder_re, ALG, RUNS, cvis)
    for i in np.arange(len(ALG)):
        NMI_values = results[i, :, :]
        table = generate_table(NMI_values, cvis, CI)
        table.to_csv(results_path + '/' + ALG[i] + '_results.csv')
    wins = count_wins(results)
    w_tb = wins_table(wins, cvis)
    w_tb.to_csv(results_path+ '/wins_results.csv')
    plot_results_seperate(cvis, results, ALG, results_path)

def plot_results_seperate(CVI_names:list, NMI:np.array, ALG_names:list, results_path:str):
    for i in np.arange(len(ALG_names)):
        plt.figure(figsize=(25,8))
        alg_nmi_values = np.mean(NMI[i,:,:], axis=0)
        bars = plt.bar(CVI_names, alg_nmi_values, color='c')
        # plt.ylim([0,1])
        plt.title('AVG ARI for each CVI using ' + ALG_names[i] + ' Clustering method')
        plt.ylabel('AVG ARI')
        plt.xlabel('CVI')
        error_min, error_max = [],[]
        for j in np.arange(len(CVI_names)):
            CI_95 = st.norm.interval(alpha=0.95, loc=alg_nmi_values[j], scale=st.sem(NMI[i, :,j]))
            error_min.append(alg_nmi_values[j] - CI_95[0])
            error_max.append(CI_95[1] - alg_nmi_values[j])
        y_error = [error_min, error_max]
        plt.bar_label(bars, padding=0, label_type='center')
        plt.errorbar(CVI_names, alg_nmi_values, yerr=y_error, c='r',fmt='.', capsize=5)
        plt.savefig(results_path + '/' + ALG_names[i] + '_barplot.png')
        plt.show()    

def plot_results_together(CVI_names:list, NMI:np.array, ALG_names:list, results_path:str):

    nmi_values = {}
    for i in np.arange(len(ALG_names)):
        alg_avg_nmi = np.mean(NMI[i, :, :], axis=0)
        nmi_values[ALG_names[i]] = alg_avg_nmi

    x = np.arange(len(CVI_names))  # the label locations
    width = 0.3  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(20,8))

    for attribute, measurement in nmi_values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('AVG NMI')
    ax.set_title('Average NMI of chosen partition according to CVIs')
    ax.set_xticks(x + width)
    ax.set_xticklabels(CVI_names)
    ax.legend(loc='upper left')
    ax.set_ylim(0, 1)
    plt.savefig(results_path + '/barplot.png') 
    plt.show()

def count_wins(NMI_results:np.array) -> np.array:
    x, y, z = NMI_results.shape
    wins = np.zeros(z)
    for i in np.arange(x):
        alg_NMI_data = NMI_results[i, :, :]
        max_vals = np.max(alg_NMI_data, axis=1)
        for j in np.arange(y):
            w = (alg_NMI_data[j, :] == max_vals[j])
            print('w = ', w)
            wins += w
            print('wins = ', wins)
    return wins

def wins_table(wins:np.array, CVI_names:list) -> pd.DataFrame:
    CVI_names = sorted(CVI_names) # incase theyre not in the correct order
    wins_data = {}
    for i in np.arange(len(CVI_names)):
        wins_data[CVI_names[i]] = wins[i]
    df = pd.DataFrame(data=wins_data, index=['Wins/Ties'])
    return df



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
                                    # alg_results.append(data['NMI']) # loading NMI
                                    alg_results.append(data['ARI']) # loading ARI 
                
                results[alg_ind, run_ind, :] = alg_results
                alg_ind += 1    

            run_ind += 1

    return results

def generate_table(CVI_values:np.array, CVI_legend:list, CI:float) -> pd.DataFrame:

    df = pd.DataFrame()
    for i in np.arange(CVI_values.shape[1]):

        # Calculate the confidence interval
        CI_95 = st.t.interval(alpha=CI,df=len(CVI_values[:,i])-1, loc=np.mean(CVI_values[:,i]), scale=st.sem(CVI_values[:,i]))
        rd = {'Mean_CVI': np.mean(CVI_values[:,i]), 'Standard_Deviation':np.std(CVI_values[:,i]), 'CI_lower':CI_95[0], 'CI_higher':CI_95[1]}  
        curr_df = pd.DataFrame(data=rd, index=[CVI_legend[i]])
        df = pd.concat([df, curr_df], ignore_index=False)    

    return df
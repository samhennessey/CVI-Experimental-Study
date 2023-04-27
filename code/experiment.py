''' 
Author: Sam Hennessey
Date: 26/4/2023
'''

from helper import *
from sklearn.metrics import normalized_mutual_info_score


PERCENTAGE_CONSTRINATS = 0.1
NMI_RESULTS_FILE = '../results/NMI_results_PC' + str(PERCENTAGE_CONSTRINATS)+ '.npy'
K_RESULTS_FILE = '../results/K_results_PC' + str(PERCENTAGE_CONSTRINATS)+ '.npy'

if __name__ == '__main__':

    # get a list of all datasets....
    datasets = get_datasets()
    
    # get a list of all algorithms.....
    ''' This get more functions than are actually there, solution could be to have a list at the top of the file that includes only the 
    functions we want access to''' 
    clMethods = get_clMethods()

    # get a list of all CVIs 
    ''' This get more functions than are actually there, solution could be to have a list at the top of the file that includes only the 
    functions we want access to'''
    cvis = get_CVIs()
    
    # store the results of the NMI of the chosen P for each dataset, each clustering method, and each CVI
    best_P_results = np.empty((len(datasets), len(clMethods), len(cvis))) 
    # store the results of the chosen K for each dataset, each clustering method, and each CVI
    best_k_results = np.empty((len(datasets), len(clMethods), len(cvis)))

    ds_ind = 0
    for ds in datasets: # for every dataset

        D, L = load_data(ds)
        N,n = D.shape
        # [ML, CL] --> GENERATE CONSTRAINTS HERE
        ML,CL = percentage_constrint_generation(D,L,PERCENTAGE_CONSTRINATS)

        K_range = np.arange(2,int(np.sqrt(N))) # range of values for K

        ds_results = np.empty((len(clMethods), len(K_range), len(cvis)))

        # store the NMI of the chosen P for each clustering method and each CVI
        best_P_ds_results = np.empty((len(clMethods), len(cvis)))

        # store the chosen k for each clustering method and each CVI
        best_k_ds_results = np.empty((len(clMethods), len(cvis)))
        
        
        clMethod_ind = 0
        for method in clMethods:
            vars = method.__code__.co_varnames # Get a list of the variables for the current method
            
            alg_results = np.empty(len(K_range), len(cvis)) # store the values of all CVIs on all p_i in P

            # Store the results of the chosen K for each CVI
            best_k_alg_results = np.empty(len(cvis))

            # Store the results for the NMI of the chosen P for each CVI
            best_P_alg_results = np.empty(len(cvis))

            partitions = np.empty(len(K_range), N)

            for K in K_range:
            
                # Generate P_i using the current method and current value of K
                if 'ML' and 'CL' in vars:
                    P = method(D, K, ML ,CL)
                else:
                    P = method(D,K)
                
                partitions[K-2,:] = P
                # Store the values of each CVI for partititon P_i
                CVI_Pi = np.empty(len(cvis))
                cvi_ind = 0
                for cvi in cvis:
                    cvi_vars = method.__code__.co_varnames
                    if 'ML' and 'CL' in cvi_vars:
                        cvi_val = cvi(D,P,ML,CL)
                    else:
                        cvi_val = cvi(D,P)

                    CVI_Pi[cvi_ind] = cvi_val
                    cvi_ind += 1
                
                
                alg_results[K-2, :] = CVI_Pi
            ds_results[clMethod_ind, :, :] = alg_results

            # get the bet partition according to the CVI values and compare with the GT
            best_P_inds = np.argmax(alg_results, axis=0)
            best_P = partitions[best_P_inds,:]
            for i in np.arange(best_P.shape[0]):
                # cvis_chosen_partition_NMI = normalized_mutual_info_score(best_P[i, :], L)
                best_P_alg_results[i] = normalized_mutual_info_score(best_P[i, :], L)
                # chosen_k = best_P_inds[i] + 2
                best_k_alg_results[i] = best_P_inds[i] + 2

            best_k_ds_results[clMethod_ind, :] = best_k_alg_results
            best_P_ds_results[clMethod_ind, :] = best_P_alg_results

            
            clMethod_ind += 1
        
        # Save the results for each dataset (# algorithms, #Partitions i.e. range of K[different for each dataset], #CVIs)
        np.save('../results/' + ds + '_results', ds_results) 

        best_k_results[ds_ind, :,:] = best_k_ds_results
        best_P_results[ds_ind, :,:] = best_P_ds_results
        ds_ind += 1

    np.save(NMI_RESULTS_FILE, best_P_results)
    np.save(K_RESULTS_FILE,best_k_results)




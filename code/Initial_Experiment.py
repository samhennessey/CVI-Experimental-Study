from CVIs import *
from methods import *
from helper import *
from randomised_normal import *
import json
from sklearn.metrics.cluster import normalized_mutual_info_score

CONSTRAINT_PERCENTAGE = 0.1
MAX_CLUSTERS = 10
RESULTS_LOCATION = './results/'

# dataset characteristics
DATA_WIDTH = 40
NO_CLUSTERS = 7
DATA_DIMS = 2

CLUSTERING_METHDOS = [K_means, single_linkage]
CVIS = [satC, satC_NH, satC_sil, satC_LCCV, satC_comb, LCCV_index, norm_sil, NH]

if __name__ == '__main__':

    # get a list of all datasets....
    # datasets = get_datasets()
    '''FOR FRANKS METHOD -> CREATE A RANGE FOR N -> [100, 1000, 100]'''
    # for ds in datasets:
    for Np in np.arange(50,1000,50):
        '''TO USE FRANKS FUNCTIONS FOR THE DATASETS GENERATOR
        
        D, L = FRANKS_METHOD(N)
        
        '''
        # D, L = load_data(ds)

        D, L = randomised_normal(DATA_WIDTH, NO_CLUSTERS, Np, DATA_DIMS)
        N,n = D.shape

        CP = int(N/0.1)

        ML, CL = constraint_generation_setN(D, L, CP) # Generate the constraints from the labels

        K_RANGE = np.arange(2, MAX_CLUSTERS + 1)

        DS_LOC = RESULTS_LOCATION + 'NP_'+ str(Np) # Save location for the current dataset -> ./results/DS.__name.__

        for cm in CLUSTERING_METHDOS:
            P_SAVE_LOC = DS_LOC + '/' + cm.__name__  # save location for the clustering method on the datasets -> ./results/DS.__name.__/cm.__name__
            cm_vars = cm.__code__.co_varnames
            CM_RESULTS = np.empty((N,len(K_RANGE))) # store the generated partions for each K for the algorithm
            for K in K_RANGE:
                if 'ML' and 'CL' in cm_vars:
                    P = cm(D, K, ML, CL)
                else:
                    P = cm(D, K)
                CM_RESULTS[:,K-2] = P
            for cvi in CVIS:
                CVI_LOC = P_SAVE_LOC + '/' + cvi.__name__ # Save the outputed values of each CVI -> ./results/DS.__name.__/cm.__name__/cvi.__name
                cvi_vars = cvi.__code__.co_varnames
                CVI_RESULTS = np.empty(len(K_RANGE))
                for K in K_RANGE:
                    if 'ML' and 'CL' in cvi_vars:
                        if 'data' not in cvi_vars:
                            cvi_val = cvi(CM_RESULTS[:,K-2], ML, CL)
                        else:
                            cvi_val = cvi(D, CM_RESULTS[:,K-2], ML, CL)
                    else:
                        cvi_val = cvi(D,CM_RESULTS[:,K-2])

                    CVI_RESULTS[K-2] = cvi_val
                
                try:
                    np.savetxt(CVI_LOC + '/cvi_values.csv', CVI_RESULTS, fmt='%f' ,delimiter=",")
                except FileNotFoundError:
                    os.makedirs(CVI_LOC + '/')
                    np.savetxt(CVI_LOC + '/cvi_values.csv', CVI_RESULTS, fmt='%f' ,delimiter=",")

                # Find the best partition for each cvi and calculate the 
                best_cvi_index = np.argmax(CVI_RESULTS)
                best_P = CM_RESULTS[:,best_cvi_index]
                NMI_partition = normalized_mutual_info_score(L.flatten(),best_P)

                partition_results = {'BEST_P':list(best_P), 'NMI':NMI_partition} 

                # Save the best partition according to the CVI value. along with the NMI of the partition with the true labels 
                with open(CVI_LOC + "/best_partition.txt", "w") as fp:
                    json.dump(partition_results, fp)  # encode dict into JSON

            np.savetxt(P_SAVE_LOC + '/partitions.csv', CM_RESULTS, fmt='%f' ,delimiter=",")

            
            
            
        
        
        

                
                
                
                 




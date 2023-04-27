import numpy as np
from sklearn.metrics import silhouette_score
from itertools import permutations
from sklearn.metrics.pairwise import euclidean_distances

def satC(P:np.array, ML:np.array,CL:np.array) -> float:
    sat = 0
    for ml in ML:
        if P[int(ml[0]-1)] == P[int(ml[1]-1)]:
            sat += 1
    for cl in CL:
        if P[int(cl[0]-1)] != P[int(cl[1]-1)]:
            sat += 1
    return sat/(ML.shape[0] + CL.shape[0])

def satC_comb(data:np.array,P:np.array,ML:np.array,CL:np.array):
    alpha = 0.6
    return alpha*satC(P,ML,CL) + ((1-alpha)/2)*NH(data,P) + ((1-alpha)/2)*norm_sil(data,P)

def satC_sil(data:np.array, P:np.array,ML:np.array,CL:np.array) -> float:
    alpha = 0.6
    return alpha*satC(P,ML,CL) + (1-alpha)*norm_sil(data,P)

def satC_NH(data:np.array,P:np.array,ML:np.array,CL:np.array):
    alpha = 0.6
    return alpha*satC(P,ML,CL) + (1-alpha)*NH(data,P)

def sil_NH(data:np.array,P:np.array):
    alpha = 0.6
    return alpha*norm_sil(data,P) + (1-alpha)*NH(data,P)

def LCCV(data:np.array,P:np.array):
    return 

def norm_sil(data:np.array, P:np.array) -> float:
    return (silhouette_score(data,P) + 1)/2

def NH(data:np.array, P:np.array) -> float:
    COM = coAssociation_matrix(P)
    dis_mat = euclidean_distances(data, data)
    coef = np.corrcoef(1-COM,dis_mat)
    return (coef + 1)/2


# Co-Association Matrix function and indicator function ....
def coAssociation_matrix(P:np.array) -> np.array:
    '''
    The coAssociation_matrix() function generates a NxN matrix of pairwise similarities between points accross all partitions in P

    Input:
        - P: type[function] All base partitons 
        
    Output:
        - type[np.array] The Co-association matrix
    '''
    N, n = P.shape
    CA_M = np.empty([N,N])
    perms = permutations(np.arange(N),2)
    for p in perms:
        i,j = p[0], p[1]
        Ci, Cj = P[i], P[j]
        CA_M[i,j] = indicator(Ci, Cj)
    CA_M = np.fill_diagonal(CA_M, 1)
    return CA_M

def indicator(Ci:int,Cj:int) -> int:
    '''The indicator() function is used as part of the co-asociation matrix to check if x_i and x_j both exist
    within the same cluster, namely Cml -> the lth cluster of partition m. It returns a 1 if both points are in the 
    cluster, and 0 otherwise
    
    Input:
        - Ci: type[int] the cluster label ith point of a partition partition
        - Cj: type[int] the cluster label jth point of a partition partition
       

        
    Output:
        - type[int] 1 if point i and j are in the same cluster, 0 otherwise
    '''

    if Ci == Cj:
        return 1
    else:
        return 0
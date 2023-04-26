import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.spatial.distance as ssd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans

''' METHODS WITHOUT CONSTRAINTS '''
def K_means(data:np.array, K:int) -> np.array:
    return KMeans(n_clusters=K, init="k-means++").fit(data).labels_

def average_linkage(data:np.array,K:int):
    dis_mat = euclidean_distances(data,data)
    X = ssd.squareform(dis_mat) # convert the redundant n*n square matrix form into a condensed nC2 array
    Z = linkage(X, method='average')
    return fcluster(Z, K, criterion='maxclust')


''' METHODS WITH CONSTAINTS '''
def PCKM(data:np.array,K:int,ML:np.array,CL:np.array) -> np.array:
    return

def CAL(data:np.array,K:int,ML:np.array,CL:np.array) -> np.array:
    return



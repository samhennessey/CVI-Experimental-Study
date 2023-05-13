import numpy as np
from sklearn.metrics import silhouette_score
from itertools import permutations
from sklearn.metrics.pairwise import euclidean_distances

from collections import Counter
import networkx as nx

ALPHA = 0.5

def satC(P:np.array, ML:np.array,CL:np.array) -> float:
    sat = 0
    for ml in ML:
        if P[int(ml[0]-1)] == P[int(ml[1]-1)]:
            sat += 1
    for cl in CL:
        if P[int(cl[0]-1)] != P[int(cl[1]-1)]:
            sat += 1
    return sat/(ML.shape[0] + CL.shape[0])

def satC_comb(data:np.array,P:np.array,ML:np.array,CL:np.array) -> float:
    return ALPHA*satC(P,ML,CL) + (1-ALPHA)*((NH(data,P) + norm_sil(data,P))/2)

def satC_sil(data:np.array, P:np.array,ML:np.array,CL:np.array) -> float:
    return ALPHA*satC(P,ML,CL) + (1-ALPHA)*norm_sil(data,P)

def satC_NH(data:np.array,P:np.array,ML:np.array,CL:np.array) -> float:
    return ALPHA*satC(P,ML,CL) + (1-ALPHA)*NH(data,P)

def sil_NH(data:np.array,P:np.array) -> float:
    return ALPHA*norm_sil(data,P) + (1-ALPHA)*NH(data,P)

def LCCV_index(data:np.array,P:np.array) -> float:
    N,n = data.shape
    dist = euclidean_distances(data,data)
    s_dist = np.sort(dist, axis=1)
    index = np.argsort(dist, axis=1)

    ### NaN-searching algorithm ###
    r = 1
    nb = np.zeros(N)
    count1 = 0
    flag = False
    RNN = np.zeros((N,N))
    while not flag:
        for i in np.arange(N):
            k = index[i, r+1]
            nb[k] += 1
            RNN[k,int(nb[k])] = i
        r += 1
        count2 = np.sum(nb == 0)
        if count2 == 0 or count1 == count2:
            flag = True
        else:
            count1 = count2

    lambda_r = r - 1
    max_nb = np.max(nb)


    # desnity of each point
    ### There is a discrepancy with the original paper! The author
    ### says that ONLY the distances of the reverse neighbours are
    ### added in the denominator. However, in their implementation,
    ### the MAX_NB are added, not just the reverse neighbours'!
    rho = np.zeros(N)
    Non = int(max_nb)
    for i in np.arange(N):
        d = np.sum(s_dist[i, 0:Non+1])
        
        rho[i] = Non/d

    ### LORE algorithm ###
    # sort the points according to the density
    ord_rho = np.argsort(rho)[::-1]

    local_core = np.zeros(N)

    for i in np.arange(N):
        p = ord_rho[i]
        neighbourhood = index[p, 0:int(nb[p])+1]

        qq = np.argmax(rho[neighbourhood])
        max_index = neighbourhood[qq]
        if local_core[max_index]== 0:
            local_core[max_index] = max_index
            
        for j in np.arange(nb[p]+1):
            j = int(j)
            if local_core[neighbourhood[j]] == 0:
                local_core[neighbourhood[j]] = local_core[max_index]
            else: # RCR rule
                q = int(local_core[neighbourhood[j]])
            
                if dist[neighbourhood[j], q] > dist[neighbourhood[j], int(local_core[max_index])]:
                    local_core[neighbourhood[j]] = local_core[max_index]
            # Determine the representative according to RTR
            for m in np.arange(N):
                if local_core[m] == neighbourhood[j]:
                    local_core[m] = local_core[neighbourhood[j]]

    # find the cores

    cores = np.unique(local_core)
    cluster_number = len(cores)
    cl = local_core.copy()

    # Graph based distances (Connectivity)
    conn = np.zeros((N,N))
    weight = np.zeros((N,N))
    for i in np.arange(N):
        for j in np.arange(1, lambda_r+1):
            x = index[i,j]
            conn[i,x] = 1/(1+dist[i,x])
            conn[x,i] = conn[i,x]
            weight[i,x] = dist[i,x]

    # comput the shortest path between cores
    shortest_path = np.zeros((cluster_number,cluster_number))
    weight2 = nx.DiGraph(weight)

    for i in np.arange(cluster_number):
        shortest_path[i,i] = 0
        for j in np.arange(i+1,cluster_number):
            
            try:
                shortest_path[i,j] = nx.dijkstra_path_length(weight2, cores[i], cores[j])
            except nx.NetworkXNoPath:
                shortest_path[i,j] = 0

            shortest_path[j,i] = shortest_path[i,j]
    max_d = np.max(shortest_path)
    shortest_path[shortest_path == 0] = max_d

    u = np.unique(P)
    new_cl = np.zeros(len(P))
    for i in np.arange(len(u)):
        new_cl[P == u[i]] = i
    P = new_cl
    nl = np.max(P)

    return computeMySWC(data,cl,nl,cores,shortest_path,local_core)

''' HELPER METHODS FOR THE LCCV CLUSTERING METHOD'''
def computeMySWC(A,cl,ncl,cores,pdist,local_core) -> float:
    ncores = len(cores)
    n, dims = A.shape
    D = np.zeros((ncores, dims))
    cl_cores = np.zeros(ncores)
    for i in np.arange(ncores):
        D[i,:] = A[cores[i], :]
        cl_cores[i] = cl[cores[i]]
    nl = np.zeros(ncores)

    # count the number of points belonging to each core
    for i in np.arange(ncores):
        for j in np.arange(n):
            if local_core[j] == cores[i] and cl[j] > 0:
                nl[i] = nl[i] + 1
    
    _, s = computeSWC(D, cl_cores, ncl, pdist)
    mcv = 0
    for i in np.arange(ncores):
        mcv = mcv+(s[i] * nl[i])
    return mcv/n

''' HELPER METHODS FOR THE LCCV CLUSTERING METHOD (SILHOUETTE+)'''
def computeSWC(D, cl, ncl, dist):
    n, _ = D.shape
    cdata = {}
    numc = ncl
    for i in np.arange(ncl):
        nump = 0
        for j in np.arange(n):
            if cl[j] == i:
                nump += 1
                cdata[i][nump,:] = D[j,:] 
                cindex[i, nump] = j
    numo = 0
    # don't compute the swc of outliers
    if np.min(cl) <= 0:
        for i in np.arange(n):
            if cl[i] <= 0:
                numo += 1
    
    swc = 0
    s1 = np.zeros(n)
    for i in np.arange(numc):
        a, b, s = [], [], []
        np, _ = cdata[i].shape
        if np > 1:
            for j in np.arange(np):
                # COMPUTE a(j)
                suma = 0
                for k in np.arange(np):
                    if j != k:
                        suma += dist[cindex[i, j], cindex[i, k]]
                a[j] = suma/(np-1)

                # COMPUTE b(j)
                d = np.ones(numc) * float('inf')
                for k in np.arange(numc):
                    if k != i:
                        np2, _ = cdata[k].shape
                        sumd = 0
                        for l in np.arange(np2):
                            sumd += dist[cindex[i,j], cindex[k,l]]
                        d[k] = sumd/np2
                b[j] = np.min(d)
                # COMPUTE s(j)
                s[j] = (b[j] - a[j])/np.max([a[j], b[j]])
                s1[cindex[i, j]] = s[j]
                swc += s[j]
    
    swc = swc/(n - numo)
    return swc, s1


def satC_LCCV(data:np.array,P:np.array, ML:np.array, CL:np.array) -> float:
    return ALPHA*satC(P,ML,CL) + (1-ALPHA)*LCCV_index(data,P)

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
import numpy as np
from sklearn.metrics import silhouette_score

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
    return alpha*satC(P,ML,CL) + ((1-alpha)/2)*NH() + ((1-alpha)/2)*norm_sil(data,P)

def satC_sil(data:np.array, P:np.array,ML:np.array,CL:np.array) -> float:
    alpha = 0.6
    return alpha*satC(P,ML,CL) + (1-alpha)*norm_sil(data,P)

def satC_NH(data:np.array,P:np.array,ML:np.array,CL:np.array):
    alpha = 0.6
    return alpha*satC(P,ML,CL) + (1-alpha)*NH()

def sil_NH(data:np.array,P:np.array):
    alpha = 0.6
    return alpha*norm_sil(data,P) + (1-alpha)*NH()

def sil_dist():
    return

def NH_dist():
    return

def satC_sil_dist():
    return

def satC_NH_dist():
    return

def norm_sil(data:np.array, P:np.array) -> float:
    return (silhouette_score(data,P) + 1)/2

def NH() -> float:
    return (... + 1)/2
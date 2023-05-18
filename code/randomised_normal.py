
def randomised_normal(w,k,N,n_dims):

    # function randomised_normal(w,k,N,n_dims)
    #
    # Function to generate randomised normal points
    # in n-dimensions
    #
    # Inputs:
    #   w: width of feature space
    #   k: no. clusters
    #   N: no. points to generate
    #   n_dims: n-dimensional space
    #
    # Output:
    #   centroids: randomised normal points, with a
    #               corresponding class label

    import math
    import numpy as np


    # get spread of N in each cluster (considering numbers that don't divide without remainders)
    T = math.ceil(N/k)
    rem = N % T
    if rem == 0:
        new_N = np.full(k,T)                    # e.g. [5,5,5,5,5] if k=5 and N=25
    else:
        new_N = np.append(np.full(k-1,T),rem)   # e.g. [6,6,6,6,2] if k=5 and N=26

    
    centroids = np.zeros(shape=(1, n_dims))
    labels = np.zeros(shape=(1,1),dtype='int')
    

    for i in range(k):
        # generate covariance matrix
        A = np.random.randn(n_dims,n_dims)
        sig = np.matmul(A,A.T)

        # random mean
        mean = np.array(np.random.rand(n_dims)*w)

        # generate and plot
        x = np.random.multivariate_normal(mean, sig, size=new_N[i],check_valid='ignore').T
        centroids = np.append(centroids,x.T,axis=0)
        
        curr_labels = np.expand_dims(np.full(np.shape(x.T)[0],i), axis=1) # However many by 1 vector of same label to correspond to data. Bloody python should go to hell with it's shapes of (n,) not being (n,1)!!! :(
        labels = np.append(labels,curr_labels,axis=0)

    # remove first "0's"
    centroids = np.delete(centroids, (0), axis=0)
    labels = np.delete(labels, (0), axis=0)

    return centroids, labels





def plot_data(centroids,labels,w,k,n_dims):

    # function plot_data(centroids,labels,w,k,n_dims)
    #
    # Function to plot the resulting data points
    # in a figure (2- and 3-D only)
    #
    # Inputs:
    #   centroids: the random normal data points
    #   labels: corresponding class labels for the
    #           above
    #   w: width of feature space
    #   k: no. clusters
    #   n_dims: n-dimensional space

    import matplotlib.pyplot as plt
    import numpy as np
    import warnings


    data = np.concatenate((centroids,labels),axis=1)

    # plots 2D data
    if n_dims == 2:
        plt.figure() # figure with axis w-by-w
        plt.xlim(0, w)
        plt.ylim(0, w)

        for i in range (k):
            plt.plot(data[data[:,2]==i,0], data[data[:,2]==i,1], '.', alpha=0.5)

        plt.show()
        plt.axis('equal')


    # plots 3D data
    elif n_dims == 3:
        plt.figure()
        ax = plt.axes(projection ="3d")

        for i in range(k):
            ax.scatter3D(data[data[:,3]==i,0], data[data[:,3]==i,1], data[data[:,3]==i,2], '.', s=2, alpha=0.5)

        plt.show()
        plt.axis('equal')


    # throws shade if it's neither
    else:
        print("Centroids: ", centroids, "Labels: ", labels)
        warnings.warn("Too many dimensions to plot. The data's still here, don't worry!")
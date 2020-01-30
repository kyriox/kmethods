import numpy as np
from scipy.linalg import svd
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state
from .sampling import Clustering
from sklearn import cluster

class Nystrom:
    def fit(self, data):
        self.data=data
        n_samples, n_features = self. data.shape
        n_components = self.n_components
        if self.algorithm in ['KMeans', 'FFT', 'Random', 'KMPP']:
            self.sampler=Clustering(algorithm=self.algorithm,n_clusters=self.n_components, 
                                    distance=self.distance).fit(self.data)
            if self.algorithm !='KMeans':
                inds=np.sort(list(self.sampler.centers_.keys()))
                sampled_features=self.data[:,inds]
            else:
                R=[]
                inds=np.sort(list(self.sampler.centroids_.keys()))
                #inds=[]
                for i in inds:
                    R.append(self.sampler.centroids_[i])
                sampled_features=np.array(R).T
        self.components_ = sampled_features
        l=min(self.components_.shape)
        print(self.components_.shape,l)
        C=self.components_
        Wi=np.linalg.pinv(C[:l,:l])
        self.K=C.dot(Wi).dot(C.T)
        return self

    def __init__(self, n_components,  kernel="linear", algorithm='Random' , distance='euclidean', **kwargs):
        self.kernel = kernel
        self.n_components = n_components
        #self.random_state = random_state
        self.algorithm=algorithm
        self.distance=distance

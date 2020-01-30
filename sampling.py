import numpy as np 
from scipy.spatial.distance import *
from scipy.stats import  pearsonr
import faiss,sys
from sklearn.cluster import KMeans as skmeans
from sklearn.metrics import pairwise_distances
#from classifiers import kNN

def  pearsond(x,y):
     return 1-np.abs(pearsonr(x,y)[0])

class Clustering:
     def _inertia(self):
        self.inertia_=0
        for j in range(len(self.data)):
             dists=[(self.distance_function(c,self.data[j],**self.kwargs),i,self.kwargs) 
                    for i,c in self.centroids_.items()]
             dists.sort()
             self.inertia_+=dists[0][0]
         
     def _find_nearest_reference(self,references):
         labels=[-1 for x in self.data]
         self.max_dist=0
         if self.algorithm.__name__=='KMeans':
             self.centers_={}
             nearest={k : np.inf for k,v in references.items()}
         for j in range(len(self.data)):
             dists=[(self.distance_function(c,self.data[j],**self.kwargs),i,self.data[j])
                    for i,c in references.items()]
             #print(dists)
             dists.sort()
             labels[j]=dists[0][1]
             if self.max_dist<dists[0][0]:
                 self.max_dist=dists[0][0]
             if self.algorithm.__name__=='KMeans' and dists[0][0]<nearest[labels[j]]:
                 self.centers_[labels[j]]=self.data[j]
                 nearest[labels[j]]=dists[0][0]
         return np.array(labels)

     def _assign_references(self):
         self.centroids_={}
         self.centers_labels_=self._find_nearest_reference(self.centers_)
         for c in self.centers_:
             self.centroids_[c]=np.mean(self.data[self.centers_labels_==c],axis=0)
         self._inertia()
         self.labels_=self._find_nearest_reference(self.centroids_)
         
     def FFT(self):
         current=np.random.randint(0, self.data.shape[0])
         dist=[self.distance_function(self.data[current], x, **self.kwargs) for x in self.data]
         ordered_args=np.argsort(dist)
         order,distances=[current],[dist[ordered_args[-1]]]
         self.centers_={current:self.data[current]}
         self.inertia_=0
         while len(self.centers_)<self.n_clusters:
             #print(order[-1],distances[-1])
             current=ordered_args[-1]
             order.append(current)
             ndist=[self.distance_function(self.data[current], x, **self.kwargs) for x in self.data]
             dist=[d if d<nd else nd for d,nd in zip(dist,ndist)]
             ordered_args=np.argsort(dist)
             distances.append(dist[ordered_args[-1]])
             self.centers_[current]=self.data[current]
         self.selection_order_=np.array(order)
         self.distances_=np.array(distances)
         self._assign_references()
         return self
         
     def add_farthest(self,data,k):
          scenters=list(self.centers_.items())
          ck=len(scenters)
          dist=[self.distance_function(scenters[0][1], x, **self.kwargs) for x in data]
          idmax=np.max([k for k,v in scenters])
          print(len(dist), np.max(dist), np.min(dist))
          for idc,c in scenters[1:]:
               ndist=[self.distance_function(c, x) for x in data]
               dist=[d if d<=nd else nd for d,nd in zip(dist,ndist)]
          ordered_args=np.argsort(dist)
          current=ordered_args[-1]
          order,distances=[current],[dist[ordered_args[-1]]]
          self.centers_={idmax+current: data[current]}
          self.inertia_=0
          self.n_clusters=self.n_clusters+k
          while len(self.centers_)<self.n_clusters:
             current=ordered_args[-1]
             order.append(current+idmax)
             ndist=[self.distance_function(data[current], x, **self.kwargs) for x in data]
             dist=[d if d<nd else nd for d,nd in zip(dist,ndist)]
             ordered_args=np.argsort(dist)
             distances.append(dist[ordered_args[-1]])
             self.centers_[current]=data[current]
          self.selection_order_=np.concatenate((self.selection_order_,order))
          self.distances_=np.concatenate((self.distances_,distances))
          #self._assign_references()
          return self
         
     def KMeans(self):
            km=skmeans(n_clusters=self.n_clusters).fit(self.data)
            self.centroids_=dict(zip(set(km.labels_),km.cluster_centers_))
            self.inertia_=km.inertia_
            #print(km.inertia_)
            self.labels_=km.labels_
            self._find_nearest_reference(self.centroids_)
            return self

     def KMPP(self):
          idx=np.random.randint(self.data.shape[0],size=1)[0]
          self.centroids_={}
          self.centers_={}
          self.labels_=[]
          self.centers_[idx]=self.data[idx]
          D=pairwise_distances(self.data,[self.data[idx]])
          for i in range(self.n_clusters-1):
               cp=np.cumsum(D/np.sum(D))
               r=np.random.rand()
               s=np.argwhere(cp>=r)[0][0]
               self.centers_[s]=self.data[s]
               dist=pairwise_distances(self.data, [self.data[s]])
               D=np.minimum(D,dist)
          self.centers_labels_=self._find_nearest_reference(self.centers_)
          for c in self.centers_:
               self.centroids_[c]=np.mean(self.data[self.centers_labels_==c],axis=0)
          self.labels_=self._find_nearest_reference(self.centroids_)
          self._inertia()
          return self

     def Random(self):
         #idx=np.random.randint(self.data.shape[0], size=self.n_clusters)
         idx=np.random.permutation(self.data.shape[0])[:self.n_clusters]
         #print(len(idx))
         self.centroids_={}
         self.labels_=[]
         centers=self.data[idx,:]
         self.centers_=dict(zip(idx,centers))
         self.centers_labels_=self._find_nearest_reference(self.centers_)
         for c in self.centers_.keys():
              self.centers_labels_[c]=c
         for c in self.centers_:
             self.centroids_[c]=np.mean(self.data[self.centers_labels_==c],axis=0)
         self.labels_=self._find_nearest_reference(self.centroids_)
         self._inertia()
         return self
     
     def _make_index(self,d):
          self.data=self.data.astype('float32')
          if self.distance=="cosine":
               vnorm=np.linalg.norm(self.data,axis=1)
               self.index=faiss.IndexFlatIP(d) # indice que utiliza el producto punto
               self.ndata=self.data/vnorm.reshape(len(vnorm),1)
          else:
               self.ndata=self.data
               self.index= faiss.IndexFlatL2(d) # indice que utiliza L2
          self.index.add(self.ndata)

     def DNet(self):
          n,d=self.data.shape
          self._make_index(d)
          self.centers_,self.centroids_={},{}
          nn=int(np.ceil(len(self.data)/self.n_clusters))
          no_selected=[i for i in range(n)]
          self.centers_labels_=np.array([-1 for i in range(n)])
          print("Bucket Size:", nn)
          while len(self.centers_)<self.n_clusters:
               i=np.random.choice(no_selected,1)[0]
               self.centers_[i]=self.data[i]
               dd,ids=self.index.search(self.ndata[i:i+1],nn)
               self.centers_labels_[ids]=i
               self.centers_labels_[i]=i
               self.index.remove_ids(ids[0])
               no_selected=np.setdiff1d(no_selected,ids)
          #if len(no_selected):
          #     self.centers_labels_[no_selected]=no_selected[0]
          #     self.centers_[no_selected[0]]=self.data[no_selected[0]] 
          for c in self.centers_:
               self.centroids_[c]=np.mean(self.data[self.centers_labels_==c],axis=0)
          #print(self.centroids_.keys(),self.centers_.keys())
          self.labels_=self._find_nearest_reference(self.centroids_)
          self.centers_labels_=self._find_nearest_reference(self.centers_)
          #self.labels_=self.centers_labels_
          return self     
         
     def fit(self,data):
         self.data=data
         #if self.distance_name=='mahalanobis':
         #    if cov is None:
         #        self.kwargs['W']=np.cov(data.T)    
         self.algorithm()
         return self
        
     def predict(self,udata, references='centers'):
        labels=[-1 for i in range(len(udata))]
        for j in range(len(udata)):
             dists=[(self.distance_function(c,udata[j],**self.kwargs),i,udata[j])
                    for i,c in getattr(self,'%s_' %references).items()]
             dists.sort()
             labels[j]=dists[0][1]
        return np.array(labels)
    
     def __init__(self,n_clusters=3,distance='euclidean',algorithm='FFT',K=None):
         self.n_clusters=n_clusters
         self.inertia_=0
         self.distance=distance
         self.distance_function=eval(distance)
         self.algorithm=getattr(self, algorithm)
         self.kwargs={}
         self.distance_name=distance

class RowSampler:
     def fit(self,data,labels):
          self.data=data
          self.labels=labels
          self.centers_={}
          self.centroids_={}
          ulabels=np.sort(np.unique(labels))
          ks={l:int(self.n_samples/len(ulabels)) for l in ulabels}
          ks[ulabels[-1]]+=self.n_samples%len(ulabels)
          if self.per_class and self.algorithm!='IFFT':
               print(ks)
               self.samplers={l:Clustering(algorithm=self.algorithm, 
                                           n_clusters=ks[l]).fit(self.data[labels==l])
                              for l in ulabels}
          else:
               if self.algorithm!='IFFT':
                    self.samplers={0: Clustering(algorithm=self.algorithm, 
                                                 n_clusters=self.n_samples).fit(self.data)}
               else:
                    ss=list(ks.items())
                    l,kl=ss[0]
                    print(ks)
                    self.samplers={0: Clustering(algorithm='FFT', 
                                                 n_clusters=kl).fit(self.data[labels==l])}
                    for l,kl in ss[1:]:
                         self.samplers[0].add_farthest(self.data[labels==l],kl)
                         
          i=0
          for l,sampler in self.samplers.items():
               for center,centroid in zip(sampler.centers_.values(),sampler.centroids_.values()):
                    self.centers_[i]=center
                    self.centroids_[i]=centroid
                    i+=1
          return self
               
     def __init__(self,n_samples=16,distance='euclidean',algorithm='FFT', features_type='centers', per_class=False):
          self.n_samples=n_samples
          self.algorithm=algorithm
          #self.classifier=classifier
          self.features_type=features_type
          self.per_class=per_class
          self.kwargs={}


class ColumnSampler:
     def fit(self,data):
          self.data=data.T
          self.sampler=Clustering(algorithm=self.algotithm, 
                                  n_clusters=self.n_samples).fit(self.data)
          return self

     def fit_transform(self,data):
          self.data=data.T
          self.sampler=Clustering(algorithm=self.algotithm, 
                                  n_clusters=self.n_samples).fit(self.data)
          if self.references_type=='centers' and algorithm!='KMeans':
               centers_ids=np.sort(list(self.sampler.centers_.keys()))
               return self.data.T[:,centers_ids]
          else:
               X=[]
               centroids_ids=np.sort(list(self.sampler.centroids_keys()))
               for i in centroids_ids:
                    X.append(self.sampler.centroids_[i])
               return np.array(X).T

     def transform(self,data):
          if self.references_type=='centers' and algorithm!='KMeans':
               centers_ids=np.sort(list(self.sampler.centers_.keys()))
               return data[:,centers_ids]
          else:
               dataT=data.T
               centroids_ids=np.sort(list(self.sampler.centroids_keys()))
               labels=self.sampler.predict(dataT,references='centroids')
               X=[]
               for i in centroids_ids:
                    X.append(np.mean(dataT[labels==i]),axis=1)
               return np.array(X).T

     def __init__(self,n_samples=3,distance='euclidean',algorithm='FFT', features_type='centers'):
          self.n_samples=n_samples
          self.algorithm=algorithm
          #self.classifier=classifier
          self.features_type=features_type
          self.kwargs={}
    
          
                                
    

         

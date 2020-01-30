import numpy as np
import faiss 
import copy
from scipy.spatial.distance import *
from sklearn.metrics.pairwise import pairwise_kernels as kernels
from .sampling import Clustering, RowSampler
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.linear_model import (LogisticRegression, SGDClassifier, 
                                  Perceptron,PassiveAggressiveClassifier)
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (recall_score, f1_score,
                             precision_score, accuracy_score)
                
class NearestCentroid:
    # Encontar el centroide más cercano
    def _find_nearest_centroid(self,sample):
        #Calcular las distancias con respecto a cada centroide
        dists=[(eval(self.distance)(c,sample),i) for i,c in self.centroids_.items()]
        #ordenar
        dists.sort()
        #regresar el indice del elemento con la menor distancia
        return dists[0][1] 
        
    # Ejemplo del la formula del promedio
    def Average(self, **kwargs):
        lb=list(set(self.labels))
        self.centroids_={}
        for j in lb:
            Gj=self.data[np.where(self.labels==j)]
            #print(lb,j,Gj.shape)
            self.centroids_[j]=np.sum(Gj,axis=0)/len(Gj)
        return self
    
    # Predecir las etiquetas para un conjunto de datos
    def predict(self,unlabeled_samples):
        y=[self._find_nearest_centroid(sample) for sample in unlabeled_samples]
        return np.array(y)
              
        
    def Sum(self,**kwarg):
        print("su implementación para la formula de suma")
        return self

    #Para Rocchio debería poder pasar los parametros beta y gamma (para eso lo **kwargs)
    def Rocchio(self,**kwargs):
         print("Su implentación para la formual de Rocchio")
      
        
    def NormSum(self,**kwarg):
        print("su implementación para la formula la suma normalizada") 

    # Metodo para entrenar el modelo, solo recibe un numpy.array con los dato de n x N.
    # Donde n es el número de elmentos y N la dimensión            
    def fit(self,data,labels,**kwargs):
        self.data=data
        self.labels=labels
        self.algorithm()
        return self

    #estructura propuesta para los algoritmos
    # La variable centroid_type es un string con el nombre de su función de calculo de centroides
    def __init__(self,distance='euclidean',centroid_type='Average', **kwargs):
        #Funcion de similitud/distancia, or defecto similitud coseno
        self.distance=distance
        self.algorithm=getattr(self, centroid_type) 

    def __str__(self):
        conf={'distance':self.distance, 'centroid_type':self.centroid_type}
        return repr(conf)
    
class kNN:
    def _uniform(self,unlabeled_samples):
        samples=unlabeled_samples
        #print("shapes Xt, X", samples.shape,self.data.shape)
        if self.distance=='cosine':
            vnorm=np.linalg.norm(samples,axis=1)
            samples=samples/vnorm.reshape(len(vnorm),1)
        #print("shapes Xt, X", samples.shape,self.data.shape)
        dists,n_ids=self.index.search(samples,self.k)
        labels=[np.argmax(np.bincount(self.labels[n_id])) for n_id in n_ids]
        return np.array(labels)

    def _mean(self,unlabeled_samples):
        samples=unlabeled_samples
        if self.distance=='cosine':
            vnorm=np.linalg.norm(samples,axis=1)
            samples=samples/vnorm.reshape(len(vnorm),1)
        dists,n_ids=self.index.search(samples,self.k)
        nnlabels=np.array([self.labels[n_id] for n_id in n_ids])
        labels=[0 for i in range(len(samples))]
        for i,di,li in zip(range(len(dists)),dists,nnlabels):
            ulabels=np.unique(li)
            res={l:np.array([0.0,0]) for l in ulabels}
            for d,l in zip(di,li):
                res[l]=res[l]+np.array([d,1])
            res=sorted(res.items(), key=lambda item: item[1][0]/item[1][1])
            labels[i]=res[0][0]
        return np.array(labels)
        
    def predict(self,unlabeled_samples):
        return self.weight(unlabeled_samples.astype('float32'))
        
    def fit(self,data,labels):
        # faiss solo acepta float32
        self.data=data.astype('float32')
        self.labels=labels
        n,d=self.data.shape
        # si se utiliza distancia coseno deben normalizarse los vectores
        if self.distance=="cosine":
            vnorm=np.linalg.norm(self.data,axis=1)
            self.index=faiss.IndexFlatIP(d) # indice que utiliza el producto punto
            self.data=self.data/vnorm.reshape(len(vnorm),1)
        else:
            self.index= faiss.IndexFlatL2(d) # indice que utiliza L2
        self.index.add(self.data) 
        return self
    
    def __init__(self,k=1,distance='cosine',weight_type='uniform'):
        #self.function=function #Funcion de similitud/distancia, por defecto similitud coseno
        self.weight=getattr(self, '_{}'.format(weight_type))
        self.distance=distance
        self.k=k

    def __str__(self):
        conf={'distance':self.distance,'number_of_neigbhors': self.k, 'weight_type':self.weight_type}
        return repr(conf)
    
class kCC:
    def fit(self, data, labels, lazy=False):
        if self.debug:
            print("Dataset (nxm)", data.shape)
            print("class disrtibution", np.bincount(labels))
        self.data=data
        self.labels=labels
        self.sampler_=RowSampler(algorithm=self.sampling,distance=self.distance, 
                                 n_samples=self.n_samples,per_class=self.per_class
        ).fit(self.data,self.labels)
        if self.debug:
            print('%s sampling finishes' %self.sampling)
            print('Applying kernel: %s' %self.kernel)
        self.centroids_=np.array(list(self.sampler_.centroids_.values()))
        if self.sampling!='KMeans':
            self.centers_=np.array(list(self.sampler_.centers_.values()))
        else:
            self.centers_=self.centroids_
        self.references=getattr(self,'%s_' %self.reference_type)
        if not lazy:  # lazy is used to prevent classiifier fit
            Xp=kernels(data,self.references, metric=self.kernel,**self.kwargs)
            self.classifier=self.classifier.fit(Xp,self.labels)
        return self

    def classifier_fit(self,data,labels):
          Xp=kernels(data,self.references, metric=self.kernel,**self.kwargs)
          self.classifier=self.classifier.fit(Xp,labels)

    def predict(self,unlabel_samples):
        Xt=kernels(unlabel_samples,self.references, metric=self.kernel,**self.kwargs)
        return self.classifier.predict(Xt)

    def transform(self,samples):
        Xt=kernels(samples,self.references, metric=self.kernel,**self.kwargs)
        return Xt
        
    def __init__(self,n_samples=2, distance='euclidean', kernel='laplacian', reference_type='centers',
                 sampling='FFT', classifier=kNN(distance='cosine', k=11),debug=False, per_class=False, **kwargs):
        print(distance, kernel, sampling)
        self.n_samples=n_samples
        self.distance=distance
        self.kernel=kernel
        self.reference_type=reference_type
        self.sampling=sampling
        self.classifier=classifier
        self.per_class=per_class
        self.kwargs=kwargs
        self.debug=debug
    
                                
class OkCC:
    #clf_list=[RandomForestClassifier(),kNN(distance='cosine', k=11), kNN(distance='euclidean', k=11),
    #          GaussianNB(),LogisticRegression(), SGDClassifier(), Perceptron(),
    #          PassiveAggressiveClassifier(), LinearSVC(), BernoulliNB(), DecisionTreeClassifier(), 
    #          ExtraTreesClassifier(), AdaBoostClassifier(), GradientBoostingClassifier()]
    #clf_list=[RandomForestClassifier(n_estimators=100),kNN(distance='cosine', k=11), 
    #          kNN(distance='euclidean', k=11),PassiveAggressiveClassifier()]
    clf_list=[kNN(distance='cosine', k=11), GaussianNB(),
              kNN(distance='euclidean', k=11),
              kNN(distance='cosine', k=5, weight_type='mean'), 
              kNN(distance='euclidean', k=5, weight_type='mean')]
              #NearestCentroid(distance='euclidean'),NearestCentroid(distance='cosine')]
              
    def __init__(self,K=[4,8,16,32,64],distances=['euclidean','cosine'], 
                 kernels_list=['linear','poly','laplacian','sigmoid','rbf','cosine'],
                 references_type=['centers','centroids'], samplings=['Random','FFT','DNet','KMeans'], 
                 classifiers=clf_list,op_function=f1_score,debug=False, sample_size=32, 
                 kfolds=3,**kwargs):
        self.K,self.distances,self.kernels_list=K,distances,kernels_list
        self.references_type,self.samplings=references_type,samplings
        self.classifiers,self.sample_size=classifiers,sample_size
        self.kwargs,self.debug=kwargs,debug
        self.kfolds,self.op_function=kfolds, op_function
        self.confs=np.array([(sampling!='KMeans' or distance!='cosine') 
                             and {'n_samples':k, 'distance': distance, 'kernel':kernel, 
                                  'reference_type':reference_type,'sampling':sampling, 
                                  'classifier':copy.deepcopy(classifier)} or None
                             for k in K for distance in distances
                             for kernel in kernels_list for reference_type in references_type
                             for sampling in samplings for  classifier in classifiers])
        self.confs=self.confs[self.confs!=None]
        self.confs=self.confs[np.random.permutation(len(self.confs))][:sample_size]
        
    def fit(self,data,labels):
        op_vals=[0 for i in range(self.sample_size)]
        self.kccs=[None for i in range(self.sample_size)]
        for i,conf in enumerate(self.confs):
            print(conf)
            skf = StratifiedKFold(n_splits=self.kfolds, random_state=33)
            skf.get_n_splits(data, labels)
            kcc=kCC(**conf,debug=True,per_class=True).fit(data,labels, lazy=True)
            avg_op_val=0
            for train_index, test_index in skf.split(data, labels):
                X_train, X_test = data[train_index], data[test_index]
                y_train, y_test = labels[train_index], labels[test_index]
                #print("yt,Xt",y_test.shape,X_test.shape)
                kcc.classifier_fit(X_train,y_train)
                avg_op_val+=self.op_function(y_test,kcc.predict(X_test))
            op_vals[i]=avg_op_val/self.kfolds
            self.kccs[i]=kcc
            print("score value: ", op_vals[i])
        self.rank=np.argsort(-1*np.array(op_vals))
        self.op_vals=op_vals
        for kcc in self.kccs:
            kcc.classifier_fit(data,labels)
    
    def predict(self,unlabeled_samples, ensemble_size=1):
        kcc=self.kccs[self.rank[0]]
        y=kcc.predict(unlabeled_samples)
        if ensemble_size==1:
            return y
        h=y[:,None]
        for k in self.rank[1:ensemble_size]:
            kcc=self.kccs[k]
            h=np.concatenate((h,kcc.predict(unlabeled_samples)[:,None]),axis=1)
        r=[np.bincount(yi, minlength=len(np.unique(kcc.labels))) for yi in h]
        #print(np.argmax(r),axis=1)
        return np.argmax(r,axis=1)
            

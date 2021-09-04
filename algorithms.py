import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class my_PCA():
    def __init__(self, n_components=None):
        self.n = n_components   

    def fit(self, X): 
        Xcentered = X - X.mean(axis = 0)
        C = Xcentered.T.dot(Xcentered) / len(X)
        self.lambd, self.F = np.linalg.eig(C)

    
    def transform(self, X): 
        Xcentered = X - X.mean(axis = 0)
        result = Xcentered.dot(self.F)
        if self.n is not None:
            self.Y = result[:,:self.n]
        else:
            self.Y = result    
    
    def fit_transform(self, X): 
        self.fit(X)
        self.transform(X)



class my_kNN:

    def __init__(self, neighbors = 5):
        self.n = neighbors
    
    def fit(self, X, labels):
        self.X = X
        self.labels = labels

    def predict(self, X):
        self.new_labels=[]
        for sample in X:
          dist = np.linalg.norm(self.X - sample,axis = 1)
          nlabels = self.labels[np.argpartition(dist,self.n)[:self.n]]
          unique, count = np.unique(nlabels, return_counts=True)
          S = dict(zip(unique, count))
          self.new_labels.append(max(S, key=S.get))
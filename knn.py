import numpy as np

#a k-nearest neighbor class in wolpertinger architecture
class knn:
    def __init__(self, cache_size, k):
        self.action_range = cache_size+1 #action range start from 0 so add 1 for np.arange
        self.k = k
        self.action_space = np.arange(self.action_range)
    
    def expand_by_KNN(self, actor):
        distance = np.power(abs(self.action_space - actor),2)
        knn_action_space = distance.argsort()[:self.k]
        return knn_action_space.reshape(self.k,1)
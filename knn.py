import numpy as np
import gc

#a k-nearest neighbor class in wolpertinger architecture
class knn:
    def __init__(self, cache_size, k, DEBUG=False):
        self.action_range = cache_size+1 #action range start from 0 so add 1 for np.arange
        self.k = k
        self.action_space = np.arange(self.action_range)
        self.DEBUG = DEBUG
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("K-nearest neighbor initialization summry:")
            print("action range = ",self.action_range)
            print("k = ",self.k)
            print("action space = ",self.action_space)
    
    def expand_by_KNN(self, actor):
        distance = np.power(abs(self.action_space - actor),2)
        knn_action_space = distance.argsort()[:self.k]
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In knn -> expand_by_KNN")
            print("distance = ",distance)
            print("knn_action_space = ",knn_action_space)
        return knn_action_space.reshape(self.k,1)
    
    def clean(self):
        del self.action_range
        del self.action_space
        del self.k
        del self.DEBUG
        gc.collect()
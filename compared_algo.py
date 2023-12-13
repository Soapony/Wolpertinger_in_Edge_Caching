import time
import numpy as np
from collections import deque
from gen_zipf import gen_zipf

class LRU:
    def __init__(self,cache_size):
        self.cache_space = [[],[]]
        self.max_size = cache_size
    
    def store(self,request_id):
        self.cache_space[0].append(request_id)
        self.cache_space[1].append(time.time_ns())
    
    def check_hit(self,request_id):
        if request_id in self.cache_space[0]:
            idx = self.cache_space[0].index(request_id)
            self.cache_space[1][idx] = time.time_ns()
            return True
        return False
    
    def replace(self,request_id):
        timestamps = np.array(self.cache_space[1])
        lru_idx = np.argmin(timestamps)
        self.cache_space[0][lru_idx] = request_id
        self.cache_space[1][lru_idx] = time.time_ns()

class LFU:
    def __init__(self,cache_size):
        self.cache_space = [[],[]]
        self.max_size = cache_size
    
    def store(self,request_id):
        self.cache_space[0].append(request_id)
        self.cache_space[1].append(0)
    
    def check_hit(self,request_id):
        if request_id in self.cache_space[0]:
            idx = self.cache_space[0].index(request_id)
            self.cache_space[1][idx] += 1
            return True
        return False
    
    def replace(self,request_id):
        frequencies = np.array(self.cache_space[1])
        lfu_idx = np.argmin(frequencies)
        self.cache_space[0][lfu_idx] = request_id
        self.cache_space[1][lfu_idx] = 0
    
class FIFO:
    def __init__(self,cache_size):
        self.cache_space = deque(maxlen = cache_size)
    
    def check_hit(self,request_id):
        if request_id in self.cache_space:
            return True
        self.cache_space.append(request_id)
        return False

if __name__ == "__main__":
    zipf = gen_zipf(1.3, 10000, 5000,False)
    requests_list = zipf.load_request("data/training_data_2varNormal2.txt")
    cache_size = 150
    lru = LRU(cache_size)
    lfu = LFU(cache_size)
    fifo = FIFO(cache_size)
    lfu_hit = 0
    lru_hit = 0
    fifo_hit = 0
    total_count = 0
    lfu_history=[]
    lru_history=[]
    fifo_history=[]
    for request_id in requests_list:
        total_count+=1

        if(lru.check_hit(request_id)):
            lru_hit+=1
        elif(len(lru.cache_space[0]) < cache_size):
            lru.store(request_id)
        else:
            lru.replace(request_id)
        
        if(lfu.check_hit(request_id)):
            lfu_hit+=1
        elif(len(lfu.cache_space[0]) < cache_size):
            lfu.store(request_id)
        else:
            lfu.replace(request_id)
        
        if(fifo.check_hit(request_id)):
            fifo_hit+=1
        
        lfu_history.append(float(lru_hit / total_count))
        lru_history.append(float(lru_hit / total_count))
        fifo_history.append(float(fifo_hit / total_count))
        
    f=open("result/compared_algo.txt","w")
    f.write(str(lfu_history)+"\n")
    f.write(str(lru_history)+"\n")
    f.write(str(fifo_history)+"\n")
    f.close()
import numpy as np
import matplotlib.pyplot as plt
import random

class gen_zipf():
    
    def __init__(self, param, size, num_files):
        self.param = param
        self.size = size
        self.num_files = num_files

    def generate_request(self,save_name = None):
        denominator_sum = sum(1.0 / (i ** self.param) for i in range(1,self.num_files+1))
        
        pdf = [(1.0 / (i ** self.param)) / denominator_sum for i in range(1,self.num_files+1)]
        pdf = np.random.permutation(pdf)

        requests = random.choices(range(self.num_files), weights=pdf, k=self.size*2)
        req1=[]
        req2=[]
        for i in range(0,len(requests),2):
            req1.append(requests[i])
        
        for i in range(1,len(requests),2):
            req2.append(requests[i])

        #save the requests in txt file
        if(save_name is not None):
            f=open(save_name+".txt", "w")
            for request in req1:
                f.write(str(request)+" ")
            f.write("\n")
            f.close()
        
        if(save_name is not None):
            f=open(save_name+"2.txt", "w")
            for request in req2:
                f.write(str(request)+" ")
            f.write("\n")
            f.close()

        return requests

    def generate_var_normal_distrib(self, new_num_files = 100, round = 3, save_name = None):
        sd = 150
        requests=[]

        for i in range(round):
            total_files = self.num_files + i * new_num_files
            mean = total_files - new_num_files / 2
            
            for j in range(8000):
                while(True):
                    tmp = np.round(np.random.normal(mean,sd,1)).astype(int)
                    if tmp[0] <= total_files:
                        requests.append(tmp[0])
                        break
        
        req1=[]
        req2=[]
        for i in range(0,len(requests),2):
            req1.append(requests[i])
        
        for i in range(1,len(requests),2):
            req2.append(requests[i])
            
        if(save_name is not None):
            f=open(save_name+".txt", "w")
            for req in req1:
                f.write(str(req)+" ")
            f.write("\n")
            f.close()
        
        if(save_name is not None):
            f=open(save_name+"2.txt", "w")
            for req in req2:
                f.write(str(req)+" ")
            f.write("\n")
            f.close()

        return requests

    def generate_2var_normal_distrib(self, save_name = None, new_num_files = 100):

        denominator_sum = sum(1.0 / (i ** self.param) for i in range(1,self.num_files+1))
        
        pdf = [(1.0 / (i ** self.param)) / denominator_sum for i in range(1,self.num_files+1)]
        pdf = np.random.permutation(pdf)

        requests = random.choices(range(self.num_files), weights=pdf, k=self.size*2)
        req1=[]
        req2=[]
        for i in range(0,len(requests),2):
            req1.append(requests[i])
        
        for i in range(1,len(requests),2):
            req2.append(requests[i])
        
        if(save_name is not None):
            f=open(save_name+".txt", "w")
            for request in req1:
                f.write(str(request)+" ")
            f.write("\n")
            f.close()
        
        
        sd = 300
        total_files = self.num_files + new_num_files
        mean = total_files - new_num_files / 2
        for i in range(5000):
            while(True):
                tmp=np.round(np.random.normal(mean,sd,1)).astype(int)
                if tmp[0]<=total_files:
                    req2.append(tmp[0])
                    break
        
        if(save_name is not None):
            f=open(save_name+"2.txt", "w")
            for request in req2:
                f.write(str(request)+" ")
            f.write("\n")
            f.close()
        
        return requests
    
    def load_request(self,file_name):
        f = open(file_name,"r")
        line = f.readline()
        requests = line.split()
        requests = [int(x) for x in requests]
        f.close()

        return requests
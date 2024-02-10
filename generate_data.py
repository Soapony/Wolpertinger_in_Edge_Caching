import numpy as np
import random

class generate_data():
    
    def __init__(self, param, size, num_files):
        self.param = param          #zipf parameter
        self.size = size            #num of request
        self.num_files = num_files  #total number of files

    def generate_zipf_pattern(self,save_name = None):
        #generate zipf distribution
        denominator_sum = sum(1.0 / (i ** self.param) for i in range(1,self.num_files+1))
        pdf = [(1.0 / (i ** self.param)) / denominator_sum for i in range(1,self.num_files+1)]
        pdf = np.random.permutation(pdf)

        #generate both offline and online requests from the distribution together
        requests = random.choices(range(self.num_files), weights=pdf, k=self.size*2)

        #separate request for offline training and online testing in an alternating manner
        req1=[]
        req2=[]
        for i in range(0,len(requests),2):
            req1.append(requests[i])
        
        for i in range(1,len(requests),2):
            req2.append(requests[i])

        #save the requests in txt file, first one for training and second one for online testing
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
    
    #generate variable normal distribution patter
    def generate_var_normal_distrib(self, new_num_files = 100, round = 3, save_name = None):
        sd = 150
        requests=[]

        #Add new files
        for i in range(round):
            total_files = self.num_files + (i+1) * new_num_files
            mean = total_files - new_num_files / 2  #shift the mean
            #generate requests for both training and testing from the distribution together
            for j in range(8000):
                #sample one request until the requested file doesn't out of total number of files
                while(True):
                    tmp = np.round(np.random.normal(mean,sd,1)).astype(int)
                    if tmp[0] <= total_files:
                        requests.append(tmp[0])
                        break
        #separate request for offline training and online testing in an alternating manner
        req1=[]
        req2=[]
        for i in range(0,len(requests),2):
            req1.append(requests[i])
        
        for i in range(1,len(requests),2):
            req2.append(requests[i])
        
        #save the requests in txt file, first one for training and second one for online testing
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

    def generate_mix_distrib(self, save_name = None, new_num_files = 100):
        #generate zipf distribution
        denominator_sum = sum(1.0 / (i ** self.param) for i in range(1,self.num_files+1))
        pdf = [(1.0 / (i ** self.param)) / denominator_sum for i in range(1,self.num_files+1)]
        pdf = np.random.permutation(pdf)
        #generate both offline and online requests from the distribution together
        requests = random.choices(range(self.num_files), weights=pdf, k=self.size*2)
        #separate request for offline training and online testing in an alternating manner
        req1=[]
        req2=[]
        for i in range(0,len(requests),2):
            req1.append(requests[i])
        
        for i in range(1,len(requests),2):
            req2.append(requests[i])
        
        #save the offline training requests in txt file.
        #As explain in disseration, the training data only contains zipf distribution pattern
        if(save_name is not None):
            f=open(save_name+".txt", "w")
            for request in req1:
                f.write(str(request)+" ")
            f.write("\n")
            f.close()
        
        #generate normal distribution
        sd = 300
        total_files = self.num_files + new_num_files
        mean = total_files - new_num_files / 2  #shift mean
        #generate requests from the distribution
        for i in range(5000):
            while(True):
                #sample one request until the requested file doesn't out of total number of files
                tmp=np.round(np.random.normal(mean,sd,1)).astype(int)
                if tmp[0]<=total_files:
                    req2.append(tmp[0])
                    break
        
        #save the online testing requests in txt file
        if(save_name is not None):
            f=open(save_name+"2.txt", "w")
            for request in req2:
                f.write(str(request)+" ")
            f.write("\n")
            f.close()
        
        return requests
    
    #load request data for simulation
    def load_request(self,file_name):
        f = open(file_name,"r")
        line = f.readline()
        requests = line.split()
        requests = [int(x) for x in requests]
        f.close()

        return requests
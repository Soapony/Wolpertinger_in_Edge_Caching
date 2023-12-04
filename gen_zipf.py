import numpy as np
import matplotlib.pyplot as plt
import random
import sys

class gen_zipf():
    
    def __init__(self, param, size, num_files, DEBUG=False):
        self.DEBUG = DEBUG
        self.param = param
        self.size = size
        self.num_files = num_files

    def generate_request(self,save_name = None):
        # Calculate the denominator sum for normalization
        denominator_sum = sum(1.0 / (i ** self.param) for i in range(1,self.num_files+1))
        
        # Generate random numbers using Zipf distribution formula
        pdf = [(1.0 / (i ** self.param)) / denominator_sum for i in range(1,self.num_files+1)]

        # Rearange the probabilities
        pdf = np.random.permutation(pdf)

        # Pickup file id by pdf
        requests = random.choices(range(self.num_files), weights=pdf, k=self.size)

        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In gen_zipf -> generate_request")
            #plot distribtuion
            count = np.bincount(requests)
            k = np.arange(max(requests)+1)
            plt.bar(k,count)
            plt.show()
            plt.close()

        #save the requests in txt file
        if(save_name is not None):
            f=open(save_name, "w")
            for request in requests:
                f.write(str(request)+" ")
            f.write("\n")
            f.close()

        return requests

    def generate_varPopulation_request(self, save_name=None):
        """
        #generate first population
        denominator_sum = sum(1.0 / (i ** self.param) for i in range(1,self.num_files+1))
        pdf = [(1.0 / (i ** self.param)) / denominator_sum for i in range(1,self.num_files+1)]
        pdf = np.random.permutation(pdf)
        requests1 = random.choices(range(self.num_files), weights=pdf, k=self.size)

        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In gen_zipf -> generate_varPopulation_request -> first population")
            #plot distribtuion
            count = np.bincount(requests1)
            k = np.arange(max(requests1)+1)
            plt.bar(k,count)
            plt.show()
            plt.clf()

        #save the requests in txt file
        if(save_name is not None):
            f=open(save_name, "w")
            for request in requests1:
                f.write(str(request)+" ")
            f.close()
        
        #generate second population
        denominator_sum = sum(1.0 / (i ** self.param) for i in range(1,self.num_files+1))
        pdf = [(1.0 / (i ** self.param)) / denominator_sum for i in range(1,self.num_files+1)]
        pdf = np.random.permutation(pdf)
        requests2 = random.choices(range(self.num_files), weights=pdf, k=self.size)

        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In gen_zipf -> generate_varPopulation_request -> second population")
            #plot distribtuion
            count = np.bincount(requests2)
            k = np.arange(max(requests2)+1)
            plt.bar(k,count)
            plt.show()
            plt.close()
        """
        sd = 400
        mean1 = 1500
        mean2 = 3500
        requests1 = []
        requests2 = []
        normal_sample1 = np.round(np.random.normal(mean1,sd,3000)).astype(int)
        for i in range(len(normal_sample1)):
            if normal_sample1[i] > self.num_files:
                normal_sample1[i] = self.num_files - (normal_sample1[i] - self.num_files)
            elif normal_sample1[i] < 0:
                normal_sample1[i] = abs(normal_sample1[i])
        requests1 = requests1 + normal_sample1.tolist()

        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In gen_zipf -> generate_varPopulation_request -> second population")
            #plot distribtuion
            count = np.bincount(requests1)
            k = np.arange(max(requests1)+1)
            plt.bar(k,count)
            plt.show()
            plt.close()

        normal_sample2 = np.round(np.random.normal(mean2,sd,7000)).astype(int)
        for i in range(len(normal_sample2)):
            if normal_sample2[i] > self.num_files:
                normal_sample2[i] = self.num_files - (normal_sample2[i] - self.num_files)
            elif normal_sample2[i] < 0:
                normal_sample2[i] = abs(normal_sample2[i])
        requests2 = requests2 + normal_sample2.tolist()

        requests = requests1 + requests2

        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In gen_zipf -> generate_varPopulation_request -> second population")
            #plot distribtuion
            count = np.bincount(requests)
            k = np.arange(max(requests)+1)
            plt.bar(k,count)
            plt.show()
            plt.close()

        #save the requests in txt file
        if(save_name is not None):
            f=open(save_name, "w")
            for request in requests:
                f.write(str(request)+" ")
            f.write("\n")
            f.close()

        return requests

    def generate_var_normal_distrib(self, new_num_files = 50, round = 3, save_name = None):
        sd = 150
        requests=[]

        for i in range(round):
            total_files = self.num_files + i * new_num_files
            mean = total_files - new_num_files / 2
            
            normal_sample = np.round(np.random.normal(mean,sd,6000)).astype(int)
            for j in range(len(normal_sample)):
                if normal_sample[j] > total_files:
                    normal_sample[j] = total_files - (normal_sample[j] - total_files)
            requests = requests + normal_sample.tolist()
        
        req1=[]
        req2=[]
        for i in range(0,len(requests),2):
            req1.append(requests[i])
        
        for i in range(1,len(requests),2):
            req2.append(requests[i])
        
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In gen_zipf -> generate_request_var_normal_distrib")
            count = np.bincount(req1)
            k = np.arange(max(req1)+1)
            plt.bar(k,count)
            plt.show()
            plt.close()
        
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In gen_zipf -> generate_request_var_normal_distrib")
            count = np.bincount(req2)
            k = np.arange(max(req2)+1)
            plt.bar(k,count)
            plt.show()
            plt.close()
            
        if(save_name is not None):
            f=open(save_name+".txt", "w")
            for req in req1:
                f.write(str(req)+" ")
            f.write("\n")
            f.close()
        
        if(save_name is not None):
            f=open(save_name+"2.txt", "w")
            for req in req1:
                f.write(str(req)+" ")
            f.write("\n")
            f.close()

        return requests

    def generate_2var_normal_distrib(self, save_name = None, new_num_files = 100):

        denominator_sum = sum(1.0 / (i ** self.param) for i in range(1,self.num_files+1))
        
        pdf = [(1.0 / (i ** self.param)) / denominator_sum for i in range(1,self.num_files+1)]
        pdf = np.random.permutation(pdf)

        requests = random.choices(range(self.num_files), weights=pdf, k=self.size)

        if self.DEBUG:
                print("============================DEBUG===================================")
                print("In gen_zipf -> generate_request_2var_normal_distrib")
                count = np.bincount(requests)
                k = np.arange(max(requests)+1)
                plt.bar(k,count)
                plt.show()
                plt.close()
        
        if(save_name is not None):
            f=open(save_name+".txt", "w")
            for request in requests:
                f.write(str(request)+" ")
            f.write("\n")
            f.close()
        
        denominator_sum = sum(1.0 / (i ** self.param) for i in range(1,self.num_files+1))
        pdf = [(1.0 / (i ** self.param)) / denominator_sum for i in range(1,self.num_files+1)]
        pdf = np.random.permutation(pdf)
        requests = []+random.choices(range(self.num_files), weights=pdf, k=self.size)
        
        
        sd = 300
        total_files = self.num_files + new_num_files
        mean = total_files - new_num_files
        normal_sample = np.round(np.random.normal(mean,sd,5000)).astype(int)
        for j in range(len(normal_sample)):
            if normal_sample[j] > total_files:
                normal_sample[j] = total_files - (normal_sample[j] - total_files)
        requests = requests + normal_sample.tolist()
        
        if self.DEBUG:
                print("============================DEBUG===================================")
                print("In gen_zipf -> generate_request_2var_normal_distrib")
                count = np.bincount(requests)
                k = np.arange(max(requests)+1)
                plt.bar(k,count)
                plt.show()
                plt.close()
        
        if(save_name is not None):
            f=open(save_name+"2.txt", "w")
            for request in requests:
                f.write(str(request)+" ")
            f.write("\n")
            f.close()
        
        return requests
    
    def load_request(self,file_name):
        f = open(file_name,"r")
        line = f.readline()
        requests = line.split()
        requests = [int(x) for x in requests]

        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In gen_zipf -> generate_request")
            print("file name = ",file_name)
            print("loaded requests:")
            print(requests)
        f.close()

        return requests


if __name__ == "__main__":
    args = sys.argv
    file_name = args[1]
    zipf = gen_zipf(0.8,5000,5000,True)
    #zipf.generate_varPopulation_request(file_name)
    #zipf.generate_2var_normal_distrib(save_name = file_name)
    zipf.generate_var_normal_distrib(save_name = file_name)
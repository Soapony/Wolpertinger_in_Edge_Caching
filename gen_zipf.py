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

        #save the requests in txt file
        if(save_name is not None):
            f=open(save_name, "a")
            for request in requests2:
                f.write(str(request)+" ")
            f.write("\n")
            f.close()

        requests = requests1 + requests2
        return requests

    def generate_var_normal_distrib(self, new_num_files = 30, round = 5, save_name = None):
        sd = 100

        requests=[]

        total_files = self.num_files
        mean = total_files - new_num_files / 2
        normal_sample = np.round(np.random.normal(mean,sd,2000)).astype(int)
        for j in range(len(normal_sample)):
            if normal_sample[j] > total_files:
                normal_sample[j] = total_files - (normal_sample[j] - total_files)
        requests = requests + normal_sample.tolist()

        for i in range(round):
            total_files = self.num_files + (i+1) * new_num_files
            mean = total_files - new_num_files / 2
            
            normal_sample = np.round(np.random.normal(mean,sd,1000)).astype(int)
            for j in range(len(normal_sample)):
                if normal_sample[j] > total_files:
                    normal_sample[j] = total_files - (normal_sample[j] - total_files)
            requests = requests + normal_sample.tolist()
            
            if self.DEBUG:
                print("============================DEBUG===================================")
                print("In gen_zipf -> generate_request_var_normal_distrib")
                count = np.bincount(normal_sample)
                k = np.arange(max(normal_sample)+1)
                plt.bar(k,count)
                plt.show()
                plt.close()
            
        if(save_name is not None):
            f=open(save_name, "w")
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
    zipf = gen_zipf(0.8,10000,5000,True)
    #zipf.generate_varPopulation_request(file_name)
    zipf.generate_var_normal_distrib(save_name = file_name)
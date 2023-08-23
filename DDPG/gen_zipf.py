import numpy as np
#import matplotlib.pyplot as plt
import random

class gen_zipf():
    
    def __init__(self, param, size, num_files, DEBUG=True):
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



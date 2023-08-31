import os

if __name__ == "__main__":
    """
    cmd = "python3 main.py 0.2"
    os.system(cmd)
    cmd = "python3 main.py 0.6"
    os.system(cmd)
    """
    #tau=[0.001,0.01,0.1,0.15]
    #knn=[0.01,0.05,0.1,0.15,0.2,0.25,0.3]
    #gamma=[0.99,0.95,0.9,0.85,0.8,0.75,0.7]
    dataset=["uniform","zipf2"]
    for i in dataset:
        #arg = str(i)
        cmd = "python3 main.py 300 paper "+i
        os.system(cmd)
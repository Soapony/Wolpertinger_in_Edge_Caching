import os
import sys

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
    args = sys.argv
    dataset = int(args[1])
    if dataset == 1:
        cmd = "rm offline_model/*"
        os.system(cmd)
        cmd = "python3 main.py 150 paper zipf train"
        os.system(cmd)
        cmd = "python3 main.py 150 paper zipf online"
        os.system(cmd)
        cmd = "python3 plotGraph.py"
        os.system(cmd)
        cmd = "mv result/hitrate_reward.png result/paper_C150_zipf_online.png"
        os.system(cmd)
        cmd = "python3 main.py 150 new zipf train"
        os.system(cmd)
        cmd = "python3 main.py 150 new zipf online"
        os.system(cmd)
        cmd = "python3 plotGraph.py"
        os.system(cmd)
        cmd = "mv result/hitrate_reward.png result/new_C150_zipf_online.png"
        os.system(cmd)
    if dataset == 2:
        cmd = "rm offline_model/*"
        os.system(cmd)
        cmd = "python3 main.py 150 new varNor train"
        os.system(cmd)
        cmd = "python3 main.py 150 new varNor online"
        os.system(cmd)
        cmd = "python3 plotGraph.py"
        os.system(cmd)
        cmd = "mv result/hitrate_reward.png result/newTanh_C150_varNor_online.png"
        os.system(cmd)
    if dataset == 3:
        cmd = "rm offline_model/*"
        os.system(cmd)
        cmd = "python3 main.py 150 paper varNor train"
        os.system(cmd)
        cmd = "python3 main.py 150 paper varNor online"
        os.system(cmd)
        cmd = "python3 main.py 150 new varNor train"
        os.system(cmd)
        cmd = "python3 main.py 150 new varNor online"
        os.system(cmd)
        cmd = "python3 plotGraph.py"
        os.system(cmd)
        cmd = "mv result/hitrate_compare.png result/C150_varNor_online.png"
        os.system(cmd)
    if dataset == 4:
        cmd = "rm offline_model/*"
        os.system(cmd)
        cmd = "python3 main.py 150 paper 2varNor train"
        os.system(cmd)
        cmd = "python3 main.py 150 paper 2varNor online"
        os.system(cmd)
        cmd = "python3 main.py 150 new 2varNor train"
        os.system(cmd)
        cmd = "python3 main.py 150 new 2varNor online"
        os.system(cmd)
        cmd = "python3 plotGraph.py"
        os.system(cmd)
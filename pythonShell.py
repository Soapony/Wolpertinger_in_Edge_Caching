import os
import sys

if __name__ == "__main__":
    args = sys.argv
    dataset = int(args[1])
    if dataset == 1:
        """
        cmd = "rm offline_model/zipf*"
        os.system(cmd)
        cmd = "python3 main.py 150 paper zipf train"
        os.system(cmd)
        cmd = "python3 main.py 150 paper zipf online"
        os.system(cmd)
        cmd = "python3 main.py 150 new zipf train"
        os.system(cmd)
        cmd = "python3 main.py 150 new zipf online"
        os.system(cmd)
        cmd = "python3 plotGraph.py 1"
        os.system(cmd)
        cmd = "mv result/hitrate_compare.png result/C150_zipf_online.png"
        os.system(cmd)
        """
        cmd = "rm offline_model/varPop*"
        os.system(cmd)
        cmd = "python3 main.py 150 paper varPop train"
        os.system(cmd)
        cmd = "python3 main.py 150 paper varPop online"
        os.system(cmd)
        cmd = "python3 main.py 150 new varPop train"
        os.system(cmd)
        cmd = "python3 main.py 150 new varPop online"
        os.system(cmd)
        cmd = "python3 plotGraph.py 1"
        os.system(cmd)
        cmd = "mv result/hitrate_compare.png result/C150_varPop_online.png"
        os.system(cmd)
        cmd = "rm offline_model/varNor*"
        os.system(cmd)
        cmd = "python3 main.py 150 paper varNor train"
        os.system(cmd)
        cmd = "python3 main.py 150 paper varNor online"
        os.system(cmd)
        cmd = "python3 main.py 150 new varNor train"
        os.system(cmd)
        cmd = "python3 main.py 150 new varNor online"
        os.system(cmd)
        cmd = "python3 plotGraph.py 1"
        os.system(cmd)
        cmd = "mv result/hitrate_compare.png result/C150_varNor_online.png"
        os.system(cmd)
        cmd = "rm offline_model/2varNor*"
        os.system(cmd)
        cmd = "python3 main.py 150 paper 2varNor train"
        os.system(cmd)
        cmd = "python3 main.py 150 paper 2varNor online"
        os.system(cmd)
        cmd = "python3 main.py 150 new 2varNor train"
        os.system(cmd)
        cmd = "python3 main.py 150 new 2varNor online"
        os.system(cmd)
        cmd = "python3 plotGraph.py 1"
        os.system(cmd)
        cmd = "mv result/hitrate_compare.png result/hitrate_respond_speed.png"
        os.system(cmd)
    if dataset == 2:
        os.system("rm *.txt")
        os.system("touch new_hitrate.txt paper_hitrate.txt")
        #reward_factor=[0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
        #knn=[0.01,0.05,0.1,0.15,0.2,0.25,0.3]
        #gamma=[0.99,0.95,0.9,0.85,0.8,0.75,0.7]
        tau=[0.001,0.01,0.1,0.15,0.2]
        for i in tau:
            os.system("rm offline_model/*")
            para = str(i)
            cmd = "python3 main.py 150 new zipf train "+para
            os.system(cmd)
            cmd = "python3 main.py 150 new zipf online "+para
            os.system(cmd)
    if dataset == 3:
        os.system("rm *.txt")
        os.system("touch new_hitrate.txt paper_hitrate.txt")
        for i in range(10):
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
        #cmd = "python3 plotGraph.py"
        #os.system(cmd)
        #cmd = "mv result/hitrate_compare.png result/C150_varNor_online.png"
        #os.system(cmd)
    if dataset == 4:
        for i in range(3):
            cmd = "python3 main.py 150 paper varPop train"
            os.system(cmd)
            cmd = "python3 main.py 150 new varPop train"
            os.system(cmd)
            cmd = "python3 main.py 150 paper varNor train"
            os.system(cmd)
            cmd = "python3 main.py 150 new varNor train"
            os.system(cmd)
            cmd = "python3 main.py 150 paper 2varNor train"
            os.system(cmd)
            cmd = "python3 main.py 150 new 2varNor train"
            os.system(cmd)
        cmd = "python3 main.py 150 paper varPop online"
        os.system(cmd)
        cmd = "python3 main.py 150 new varPop online"
        os.system(cmd)
        cmd = "python3 plotGraph.py 1"
        os.system(cmd)
        cmd = "mv result/hitrate_compare.png result/C150_varPop_online.png"
        os.system(cmd)
        cmd = "python3 main.py 150 paper varNor online"
        os.system(cmd)
        cmd = "python3 main.py 150 new varNor online"
        os.system(cmd)
        cmd = "python3 plotGraph.py 1"
        os.system(cmd)
        cmd = "mv result/hitrate_compare.png result/C150_varNor_online.png"
        os.system(cmd)
        cmd = "python3 main.py 150 paper 2varNor online"
        os.system(cmd)
        cmd = "python3 main.py 150 new 2varNor online"
        os.system(cmd)
        cmd = "python3 plotGraph.py 1"
        os.system(cmd)
        cmd = "mv result/hitrate_compare.png result/hitrate_respond_speed.png"
        os.system(cmd)
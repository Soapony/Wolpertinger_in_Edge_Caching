#!!!this python script is just an automation for running the program. Not related to the proposed algorithm!!!
import os
import sys

if __name__ == "__main__":
    args = sys.argv
    dataset = int(args[1])
        
    if dataset == 4:
        for i in range(15):
            cmd = "python3 main.py 150 paper zipf train"
            os.system(cmd)
            cmd = "python3 main.py 150 new zipf train"
            os.system(cmd)
            #cmd = "python3 main.py 150 paper varNor train"
            #os.system(cmd)
            #cmd = "python3 main.py 150 new varNor train"
            #os.system(cmd)
            #cmd = "python3 main.py 150 paper 2varNor train"
            #os.system(cmd)
            #cmd = "python3 main.py 150 new 2varNor train"
            #os.system(cmd)
        
        cmd = "python3 main.py 150 paper zipf online"
        os.system(cmd)
        cmd = "python3 main.py 150 new zipf online"
        os.system(cmd)
        cmd = "python3 plotGraph.py"
        os.system(cmd)
        cmd = "mv result/hitrate_compare.png result/C150_zipf_online.png"
        os.system(cmd)
        """
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
        """
    if dataset == 2:
        for i in range(3):
            cmd = "python3 main.py 150 original zipf offline"
            os.system(cmd)
            cmd = "python3 main.py 150 proposed zipf offline"
            os.system(cmd)
        cmd = "python3 main.py 150 original zipf online"
        os.system(cmd)
        cmd = "python3 main.py 150 proposed zipf online"
        os.system(cmd)
        cmd = "python3 plotGraph.py"
        os.system(cmd)
    
    if dataset == 1:
        for i in range(3):
            cmd = "python3 main.py 150 proposed varNor offline"
            os.system(cmd)
            cmd = "python3 main.py 150 original varNor offline"
            os.system(cmd)
        cmd = "python3 main.py 150 original varNor online"
        os.system(cmd)
        cmd = "python3 main.py 150 proposed varNor online"
        os.system(cmd)
        cmd = "python3 plotGraph.py"
        os.system(cmd)

    if dataset == 3:
        cmd = "python3 main.py 150 original varNor online"
        os.system(cmd)
        cmd = "python3 main.py 150 proposed varNor online"
        os.system(cmd)
        cmd = "python3 plotGraph.py"
        os.system(cmd)
        


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
        cmd = "python3 plotGraph.py 1"
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
        cmd = "python3 main.py 150 paper zipf online"
        os.system(cmd)
        cmd = "python3 main.py 150 new zipf online"
        os.system(cmd)
        cmd = "python3 plotGraph.py 1"
        os.system(cmd)
    
    if dataset == 3:
        cmd = "rm new_hitrate.txt paper_hitrate.txt"
        os.system(cmd)
        cmd = "touch new_hitrate.txt paper_hitrate.txt"
        os.system(cmd)
        for i in range(10):
            cmd = "python3 main.py 150 paper zipf online"
            os.system(cmd)
            cmd = "python3 main.py 150 new zipf online"
            os.system(cmd)
        


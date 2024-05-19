#!!!this python script is just an automation for running the program. Not related to the proposed algorithm!!!
import os
import sys

if __name__ == "__main__":
    args = sys.argv
    dataset = int(args[1])
        
    if dataset == 4:
        for i in range(3):
            cmd = "python3 main.py 150 proposed mix offline"
            os.system(cmd)
            cmd = "python3 main.py 150 original mix offline"
            os.system(cmd)
        cmd = "python3 main.py 150 original mix online"
        os.system(cmd)
        cmd = "python3 main.py 150 proposed mix online"
        os.system(cmd)
        cmd = "python3 plotGraph.py"
        os.system(cmd)
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
        


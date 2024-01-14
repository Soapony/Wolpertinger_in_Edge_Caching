import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    f = open("result/paper_hit_history.txt","r")
    f2 = open("result/new_hit_history.txt","r")
    lines = f.readlines()
    lines2 = f2.readlines()
    history = []
    history2 = []
    for line in lines:
        length = len(line)
        tmp = line[1:length-2]
        tmp = tmp.split(", ")
        tmp = [float(x) for x in tmp]
        history = history + tmp
    for line in lines2:
        length = len(line)
        tmp = line[1:length-2]
        tmp = tmp.split(", ")
        tmp = [float(x) for x in tmp]
        history2 = history2 + tmp
    history = history[1000:]
    history2 = history2[1000:]
    x = np.arange(len(history))
    x = x+1000
    plt.plot(x,history,label="original framework")
    plt.plot(x,history2,label="proposed algorithm")
    plt.legend()
    plt.savefig("result/hitrate_compare.png")
    plt.close()
    f.close()
    f2.close()
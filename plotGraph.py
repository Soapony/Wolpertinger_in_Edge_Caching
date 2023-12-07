import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":
    args = sys.argv
    expr = int(args[1])
    if expr == 1:
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
        plt.plot(x,history,label="original framwork")
        plt.plot(x,history2,label="proposed algorithm")
        plt.legend()
        plt.savefig("result/hitrate_compare.png")
        plt.close()
        f.close()
        f2.close()
    elif expr == 2:
        f = open("result/new_hit_history.txt","r")
        f2 = open("result/compared_algo.txt","r")
        drl = f.readlines()
        lfu = f2.readline()
        lru = f2.readline()
        fifo = f2.readline()
        history = []
        history_lru = []
        history_lfu = []
        history_fifo = []
        for i in drl:
            length = len(i)
            tmp = i[1:length-2]
            tmp = tmp.split(", ")
            tmp = [float(x) for x in tmp]
            history = history + tmp
        length = len(lfu)
        tmp = lfu[1:length-2]
        tmp = tmp.split(", ")
        tmp = [float(x) for x in tmp]
        history_lfu = history_lfu + tmp
        length = len(lru)
        tmp = lru[1:length-2]
        tmp = tmp.split(", ")
        tmp = [float(x) for x in tmp]
        history_lru = history_lru + tmp
        length = len(fifo)
        tmp = fifo[1:length-2]
        tmp = tmp.split(", ")
        tmp = [float(x) for x in tmp]
        history_fifo = history_fifo + tmp
        
        history = history[1000:]
        history_lfu = history_lfu[1000:]
        history_lru = history_lru[1000:]
        history_fifo = history_fifo[1000:]
        x = np.arange(len(history))
        x = x+1000
        plt.plot(x,history,label="proposed algorithm")
        plt.plot(x,history_lfu,label="LFU")
        plt.plot(x,history_lru,label="LRU")
        plt.plot(x,history_fifo,label="FIFO")
        plt.legend()
        plt.savefig("result/hitrate_compare2.png")
        plt.close()
        f.close()
        f2.close()
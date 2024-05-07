import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

#plot the graph of two approaches' cahce hit rates in the whole online phase
if __name__ == "__main__":
    #read and process the data
    f = open("result/original_hit_history.txt","r")
    f2 = open("result/proposed_hit_history.txt","r")
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
    
    #clip the range from 1000th request to the end
    history = history[1000:]
    history2 = history2[1000:]
    x = np.arange(len(history))
    x = x+1000
    #plot the cache hit rate by matplotlib.pyplot module
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.plot(x,history,label="original framework")
    plt.plot(x,history2,label="proposed algorithm")
    plt.legend()
    plt.savefig("result/hitrate_compare.png")
    plt.close()
    f.close()
    f2.close()
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    """
    f = open("result/find_reward_dis_fac.txt","r")
    lines = f.readlines()
    results=[]
    for line in lines:
        length = len(line)
        line = line[6:length-2]
        line = line.split(", ")
        line = [float(x) for x in line]
        results.append(line)
    x = [i for i in range(7)]
    i=0.1
    for result in results:
        plt.plot(x,result,label=str(i))
        i+=0.1
    plt.legend()
    #plt.show()
    plt.savefig("result/reward_dis_fac.png")
    plt.close()
    f.close()
    f = open("result/find_knn_ratio.txt","r")
    lines = f.readlines()
    results=[]
    for line in lines:
        length = len(line)
        line = line[7:length-2]
        line = line.split(", ")
        line = [float(x) for x in line]
        results.append(line)
    x = [i for i in range(7)]
    i=0
    knn=[0.01,0.05,0.1,0.15,0.2,0.25,0.3]
    for result in results:
        plt.plot(x,result,label=str(knn[i]))
        i+=1
    plt.legend()
    #plt.show()
    plt.savefig("result/knn_fraction.png")
    plt.close()
    f.close()
    f = open("result/find_ddpg_gamma.txt","r")
    lines = f.readlines()
    results=[]
    for line in lines:
        length = len(line)
        line = line[7:length-2]
        line = line.split(", ")
        line = [float(x) for x in line]
        results.append(line)
    x = [i for i in range(12)]
    i=0.5
    for result in results:
        plt.plot(x,result,label=str(i))
        i+=0.05
    plt.legend()
    #plt.show()
    plt.savefig("result/ddpg_gamma.png")
    plt.close()
    f.close()
    f = open("result/find_tau.txt","r")
    lines = f.readlines()
    results=[]
    for line in lines:
        length = len(line)
        line = line[8:length-2]
        line = line.split(", ")
        line = [float(x) for x in line]
        results.append(line)
    x = [i for i in range(7)]
    labe=[0.001,0.01,0.1,0.15]
    i=0
    for result in results:
        plt.plot(x,result,label=labe[i])
        i+=1
    plt.legend()
    #plt.show()
    plt.savefig("result/ddpg_tau.png")
    plt.close()
    f.close()
    """
    """
    f = open("result/hit_history.txt","r")
    figure, axis = plt.subplots(1,2,figsize=(15,10))
    lines = f.readlines()
    history = []
    for line in lines:
        length = len(line)
        tmp = line[1:length-2]
        tmp = tmp.split(", ")
        tmp = [float(x) for x in tmp]
        history = history + tmp
    x = np.arange(len(history))
    axis[0].plot(x,history,label="hit_history")
    axis[0].legend()
    f.close()
    f1 = open("result/act_reward.txt","r")
    f2 = open("result/pre_reward.txt","r")
    lines = f1.readlines()
    actual_reward = []
    for line in lines:
        length = len(line)
        tmp = line[1:length-2]
        tmp = tmp.split(", ")
        tmp = [float(x) for x in tmp]
        actual_reward = actual_reward + tmp
    x = np.arange(len(actual_reward))
    axis[1].plot(x,actual_reward,label="actual")
    lines = f2.readlines()
    predict_reward = []
    for line in lines:
        length = len(line)
        tmp = line[1:length-2]
        tmp = tmp.split(", ")
        tmp = [float(x) for x in tmp]
        predict_reward = predict_reward + tmp
    x = np.arange(len(predict_reward))
    axis[1].plot(x,predict_reward,label="predict")
    axis[1].legend()
    plt.savefig("result/hitrate_reward.png")
    plt.close()
    f1.close()
    f2.close()
    """
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
    plt.plot(x,history,label="paper_hit_history")
    plt.plot(x,history2,label="proposed_hit_history")
    plt.legend()
    plt.savefig("result/hitrate_compare.png")
    plt.close()
    f.close()
    f2.close()

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
    x = [i for i in range(12)]
    i=0.1
    for result in results:
        plt.plot(x,result,label=str(i))
        i+=0.1
    plt.legend()
    #plt.show()
    plt.savefig("result/reward_dis_fac.png")
    plt.close()
    f.close()
    f = open("result/find_knn_fraction.txt","r")
    lines = f.readlines()
    results=[]
    for line in lines:
        length = len(line)
        line = line[7:length-2]
        line = line.split(", ")
        line = [float(x) for x in line]
        results.append(line)
    x = [i for i in range(12)]
    i=0.1
    for result in results:
        plt.plot(x,result,label=str(i))
        i+=0.05
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
    f = open("result/find_ddpg_tau.txt","r")
    lines = f.readlines()
    results=[]
    for line in lines:
        length = len(line)
        line = line[8:length-2]
        line = line.split(", ")
        line = [float(x) for x in line]
        results.append(line)
    x = [i for i in range(10)]
    labe=["0.001","0.005","0.01","0.015","0.05","0.1","0.15","0.2"]
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
    f = open("result/reward_error.txt","r")
    lines = f.readlines()
    results=[]
    for line in lines:
        length = len(line)
        line = line[1:length-2]
        line = line.split(", ")
        line = [float(x) for x in line]
        results.append(line)
    i=0
    ax = [[0,1,0,1,0,1],[0,0,1,1,2,2]]
    figure, axis = plt.subplots(3,2,figsize=(30,20))
    labels = ["6 features", "8 features", "3layers", "4layers_dec", "5layers","6layers_dec"]
    for result in results:
        x = np.arange(len(result))
        axis[ax[1][i],ax[0][i]].plot(x,result,label=labels[i])
        axis[ax[1][i],ax[0][i]].legend()
        i+=1
    #plt.show()
    plt.savefig("result/reward_error.png")
    plt.close()
    """
    f = open("result/reward_error.txt","r")
    figure, axis = plt.subplots(1,2,figsize=(15,10))
    line = f.readline()
    length = len(line)
    line = line[1:length-2]
    line = line.split(", ")
    line = [float(x) for x in line]
    x = np.arange(len(line))
    axis[0].plot(x,line,label="reward_error")
    axis[0].legend()
    f.close()
    f2 = open("result/pre_act_reward.txt","r")
    lines = f2.readlines()
    predict_reward = lines[0]
    length = len(predict_reward)
    predict_reward = predict_reward[1:length-2]
    predict_reward = predict_reward.split(", ")
    predict_reward = [float(x) for x in predict_reward]
    x = np.arange(len(predict_reward))
    axis[1].plot(x,predict_reward,label="predict")
    actual_reward = lines[1]
    length = len(actual_reward)
    actual_reward = actual_reward[1:length-2]
    actual_reward = actual_reward.split(", ")
    actual_reward = [float(x) for x in actual_reward]
    x = np.arange(len(actual_reward))
    axis[1].plot(x,actual_reward,label="actual")
    axis[1].legend()
    plt.savefig("result/reward_error.png")
    plt.close()
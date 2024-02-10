import numpy as np
from scipy import stats
import sys

if __name__ == "__main__":
    args = sys.argv
    dataset = args[1]
    #load the cache hit rates from original framework
    if dataset == "zipf":
        f=open("result/original_zipf_hitrate.txt","r")
    elif dataset == "varNor":
        f=open("result/original_varNor_hitrate.txt","r")
    elif dataset == "mix":
        f=open("result/original_mix_hitrate.txt","r")
    else:
        print("error args")
    
    line=f.readline()
    original_hitrate = line.split(" ")
    original_hitrate = original_hitrate[:-1]
    original_hitrate = np.array([float(x) for x in original_hitrate])
    f.close()

    #load the cache hit rates from proposed method
    if dataset == "zipf":
        f=open("result/proposed_zipf_hitrate.txt","r")
    elif dataset == "varNor":
        f=open("result/proposed_varNor_hitrate.txt","r")
    elif dataset == "mix":
        f=open("result/proposed_mix_hitrate.txt","r")
    else:
        print("error args")

    line=f.readline()
    proposed_hitrate = line.split(" ")
    proposed_hitrate = proposed_hitrate[:-1]
    proposed_hitrate = np.array([float(x) for x in proposed_hitrate])
    f.close()

    #calculate the p_value by scipy.stats module
    t_statistic, p_value = stats.ttest_ind(original_hitrate,proposed_hitrate,alternative='less')
    print("T-statstic:", t_statistic)
    print("P-value:",p_value)
    if p_value < 0.05:
        print("reject the null hypothesis that paper's sample mean is greater or equal to proposed's sample mean, alpha is bigger than p_value in 95% confidence level")

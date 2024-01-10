import numpy as np
from scipy import stats

if __name__ == "__main__":
    #f=open("result/paper_zipf_hitrate.txt","r")
    #f=open("result/paper_varPop_hitrate.txt","r")
    f=open("result/paper_hitrate.txt","r")
    line=f.readline()
    paper_hitrate = line.split(" ")
    paper_hitrate = paper_hitrate[:-1]
    paper_hitrate = np.array([float(x) for x in paper_hitrate])
    f.close()

    #f=open("result/new_zipf_hitrate.txt","r")
    #f=open("result/new_varPop_hitrate.txt","r")
    f=open("new_hitrate.txt","r")
    line=f.readline()
    new_hitrate = line.split(" ")
    new_hitrate = new_hitrate[:-1]
    new_hitrate = np.array([float(x) for x in new_hitrate])
    f.close()

    alpha = 0.05
    t_statistic, p_value = stats.ttest_ind(paper_hitrate,new_hitrate,alternative='less')
    print("T-statstic:", t_statistic)
    print("P-value:",p_value)
    if alpha > p_value:
        print("reject the null hypothesis that paper's sample mean is greater or equal to proposed's sample mean, alpha is bigger than p_value in 95% confidence level")

    difference = new_hitrate - paper_hitrate
    print(difference)
    null_hypothesis_mean = 0
    t_statistic, p_value = stats.ttest_1samp(a=difference, popmean=null_hypothesis_mean)
    
    print("T-statistic:", t_statistic)
    print("P-value:",p_value)
    if alpha > p_value:
        print("reject the null hypothesis that the mean of proposed sample minus paper sample is 0, alpha is bigger than p_value in 95% confidence level")
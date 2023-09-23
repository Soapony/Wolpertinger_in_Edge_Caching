import numpy as np
from scipy import stats

if __name__ == "__main__":
    f=open("result/paper_zipf_hitrate.txt","r")
    line=f.readline()
    paper_hitrate = line.split(" ")
    paper_hitrate = paper_hitrate[:-1]
    paper_hitrate = np.array([float(x) for x in paper_hitrate])
    f.close()

    f=open("result/new_zipf_hitrate.txt","r")
    line=f.readline()
    new_hitrate = line.split(" ")
    new_hitrate = new_hitrate[:-1]
    new_hitrate = np.array([float(x) for x in new_hitrate])
    f.close()

    difference = new_hitrate - paper_hitrate
    print(difference)
    null_hypothesis_mean = 0.
    t_statistic, p_value = stats.ttest_1samp(difference, null_hypothesis_mean)
    degrees_of_freedom = len(difference) - 1
    confidence_level = 0.95
    alpha = 1 - confidence_level
    critical_t_value = stats.t.ppf(confidence_level, df=degrees_of_freedom)

    if t_statistic > critical_t_value:
        print("Reject the null hypothesis. Mean is greater than 0.")
    else:
        print("Fail to reject the null hypothesis. Mean is less than or equal to 0.")
    
    print("T-statistic:", t_statistic)
    print("P-value:",p_value)
    print("Critical t-value:", critical_t_value)
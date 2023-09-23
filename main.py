from wolpertinger import wolpertinger
from cache_env import cache_env
from gen_zipf import gen_zipf
import sys
import gc

def offline(cache_size, model, dataset):
    max_episodes = 10
    tau = 0.001
    knn = 0.2
    reward_fac = 0.9
    gamma = 0.9
    zipf = gen_zipf(1.3, 10000, 5000)
    if dataset == "zipf":
        requests_list = zipf.load_request("data/training_data.txt")
    elif dataset == "varPop":
        requests_list = zipf.load_request("data/training_data_varPopulation.txt")
    elif dataset == "varNor":
        requests_list = zipf.load_request("data/training_data_varNormal.txt")
    elif dataset == "2varNor":
        requests_list = zipf.load_request("data/training_data_2varNormal.txt")
    else:
        print("error args")
        return
    
    env = cache_env(cache_size, requests_list, model, False, reward_fac)
    drl_wol = wolpertinger(env, cache_size, model, False, knn, gamma, tau)
    hit_rate = drl_wol.offline_train(max_episodes)

    env.clean()
    drl_wol.clean()
    del env
    del drl_wol
    gc.collect()

    print("------------------------DEBUG--------------------------")
    print("hit rate: ", hit_rate)

    return

def online(cache_size,model,dataset):
    tau = 0.001
    knn = 0.2
    reward_fac = 0.9
    gamma = 0.9
    zipf = gen_zipf(1.3, 10000, 5000)
    if dataset == "varPop":
        requests_list = zipf.load_request("data/training_data_varPopulation2.txt")
    elif dataset == "varNor":
        requests_list = zipf.load_request("data/training_data_varNormal2.txt")
    elif dataset == "zipf":
        requests_list = zipf.load_request("data/training_data2.txt")
    elif dataset == "2varNor":
        requests_list = zipf.load_request("data/training_data_2varNormal2.txt")
    else:
        return
    
    env = cache_env(cache_size, requests_list, model, False, reward_fac,True)
    drl_wol = wolpertinger(env, cache_size, model, False, knn, gamma, tau)
    hit_rate = drl_wol.online_learning()
    f=open("result.txt","a")
    f.write(str(hit_rate)+" ")
    f.close()
    return

if __name__ == "__main__":
    args = sys.argv
    cache_size = int(args[1])
    model = args[2]
    dataset = args[3]
    mode = args[4]

    if mode == "train":
        #run training
        offline(cache_size,model,dataset)
    elif mode == "online":
        online(cache_size,model,dataset)
    else:
        print("error args")
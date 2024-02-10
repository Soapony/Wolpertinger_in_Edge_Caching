from wolpertinger import wolpertinger
from cache_env import cache_env
from generate_data import generate_data
import sys

#offline phase training
def offline(cache_size, model, dataset):
    max_episodes = 15
    tau = 0.1
    knn = 0.1
    reward_fac = 0.9
    gamma = 0.99
    gen_data = generate_data(1.3, 10000, 5000)
    #load requests
    if dataset == "zipf":
        requests_list = gen_data.load_request("data/request_data_zipf.txt")
    elif dataset == "varNor":
        requests_list = gen_data.load_request("data/request_data_varNormal.txt")
    elif dataset == "mix":
        requests_list = gen_data.load_request("data/request_data_mix.txt")
    else:
        print("error args")
        return
    #initilize environment and DRL agent
    env = cache_env(cache_size, requests_list, model, reward_fac)
    drl_wol = wolpertinger(env, cache_size, model, knn, gamma, tau)
    #start training
    drl_wol.offline_train(max_episodes)
    return

#online phase testing
def online(cache_size,model,dataset):
    tau = 0.1
    knn = 0.1
    reward_fac = 0.9
    gamma = 0.99
    gen_data = generate_data(1.3, 10000, 5000)
    if dataset == "varNor":
        requests_list = gen_data.load_request("data/request_data_varNormal2.txt")
    elif dataset == "zipf":
        requests_list = gen_data.load_request("data/request_data_zipf2.txt")
    elif dataset == "mix":
        requests_list = gen_data.load_request("data/request_data_mix2.txt")
    else:
        print("error args")
        return
    #initilize environment and DRL agent
    env = cache_env(cache_size, requests_list, model, reward_fac)
    drl_wol = wolpertinger(env, cache_size, model, knn, gamma, tau)
    #start online phase
    hit_rate = drl_wol.online_learning()
    #save the final cache hit rate
    if model == "original":
        f=open("original_hitrate.txt","w")
        f.write(str(hit_rate)+" ")
        f.close()
    else:
        f=open("proposed_hitrate.txt","w")
        f.write(str(hit_rate)+" ")
        f.close()
    return

if __name__ == "__main__":
    args = sys.argv
    cache_size = int(args[1])
    model = args[2]     #select the DRL agent from original framework or proposed method
    dataset = args[3]   #select request pattern
    phase = args[4]     #run in offline or online phase

    if phase == "offline":
        offline(cache_size,model,dataset)
    elif phase == "online":
        online(cache_size,model,dataset)
    else:
        print("error args")
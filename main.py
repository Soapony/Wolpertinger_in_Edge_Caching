from wolpertinger import wolpertinger
from cache_env import cache_env
from gen_zipf import gen_zipf
import sys
import gc

def do_one_train(cache_size, model, dataset):
    max_episodes = 10
    tau = 0.001
    knn = 0.2
    reward_fac = 0.9
    gamma = 0.9
    zipf = gen_zipf(1.3, 10000, 5000)
    if dataset == "zipf1":
        requests_list = zipf.load_request("training_data.txt")
    elif dataset == "uniform":
        requests_list = zipf.load_request("training_data_uniform.txt")
    elif dataset == "zipf2":
        requests_list = zipf.load_request("training_data2.txt")
    else:
        return 0.0
    #requests_list = zipf.load_request("training_data_varNormal.txt")
    #requests_list = zipf.load_request("training_data_varPopulation.txt")
    env = cache_env(cache_size, requests_list, model, False, reward_fac)
    drl_wol = wolpertinger(env, cache_size, model, False, knn, gamma, tau)
    hit_rate = drl_wol.offline_train(max_episodes)
    
    #requests_list = zipf.load_request("training_data_varPopulation.txt")
    #env = cache_env(cache_size, requests_list, model, False, reward_fac)
    #hit_rate = drl_wol.online_learning(env)

    env.clean()
    drl_wol.clean()
    del env
    del drl_wol
    gc.collect()

    return hit_rate

if __name__ == "__main__":
    args = sys.argv
    cache_size = int(args[1])
    model = args[2]
    dataset = args[3]

    #run training
    hit_rate = do_one_train(cache_size,model,dataset)
    print("------------------------DEBUG--------------------------")
    print("hit rate: ", hit_rate)
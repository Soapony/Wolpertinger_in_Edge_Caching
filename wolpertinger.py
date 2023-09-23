from ddpg import ddpg
from knn import knn
import gc

#this class implements wolpertinger architecture
class wolpertinger:
    def __init__(self, cache_env, cache_size, model="new", DEBUG=False, KNN_fraction = 0.2, gamma = 0.9, tau=0.001):
        self.env = cache_env
        self.KNN_fraction = KNN_fraction
        self.cache_size = cache_size
        self.K = round((self.cache_size+1) * self.KNN_fraction)
        self.state_shape = self.env.get_state_shape()
        self.DEBUG = DEBUG
        self.predict_rewards=[]
        self.actual_rewards=[]
        self.tau = tau
        self.model = model

        self.ddpg = ddpg(self.state_shape, self.cache_size, model, self.DEBUG, gamma, self.tau)
        self.knn = knn(self.cache_size, self.K,self.DEBUG)

        if self.DEBUG:
            print("============================DEBUG===================================")
            print("wolpertinger initialization summry:")
            print("KNN fraction = ",self.KNN_fraction)
            print("cache size = ",self.cache_size)
            print("K = ",self.K)
            print("state shape = ",self.state_shape)
    
    def offline_train(self, max_episodes):
        episodes, total_rewards, done = 0, 0.0, False
        #state is flatten and state shape is (3C,)
        current_state, done = self.env.reset()

        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In wolpertinger -> offline_train -> after env.reset() -> current_state:")
            print(current_state)
            print("done = ",done)

        while(episodes < max_episodes):
            if self.DEBUG:
                print("============================DEBUG===================================")
                print("In wolpertinger -> offline_train -> episodes:",episodes)
            if done:
                episodes +=1
                cur_hit_rate = self.env.get_hit_rate()
                print("episode:"+str(episodes)+" total rewards:"+str(total_rewards)+" hit rate:"+str(cur_hit_rate))

                self.env.hit_history.clear()
                total_rewards, done = 0.0, False
                current_state, done = self.env.reset()
                continue
            
            if self.DEBUG:
                print("============================DEBUG===================================")
                print("current state:")
                print(current_state)
                
            proto_actor = self.ddpg.actor.get_proto_actor(current_state)
            if self.DEBUG:
                print("============================DEBUG===================================")
                print("In wolpertinger -> offline_train -> proto_actor:",proto_actor)

            knn_action_space = self.knn.expand_by_KNN(proto_actor)
            if self.DEBUG:
                print("============================DEBUG===================================")
                print("In wolpertinger -> offline_train -> knn_action_space:")
                print(knn_action_space)

            index, _ = self.ddpg.critic.get_best_q_value_and_action_index(current_state, knn_action_space)
            best_action = int(knn_action_space[index])
            if self.DEBUG:
                print("============================DEBUG===================================")
                print("In wolpertinger -> offline_train -> index & best_action:",index,best_action)

            next_state, reward, done = self.env.step(best_action)

            if self.DEBUG:
                print("============================DEBUG===================================")
                print("In wolpertinger -> offline_train -> after env.step()")
                print("next_state:")
                print(next_state)
                print("reward = ",reward,"done = ",done)

            self.ddpg.brain.remember(current_state, best_action, reward, next_state, done)
            self.ddpg.replay()

            current_state = next_state
            total_rewards += reward
            if self.DEBUG:
                print("============================DEBUG===================================")
                print("In wolpertinger -> offline_train -> end of while loop")
                print("new state:")
                print(current_state)
                print("total rewards = ",total_rewards)

        if self.model == "paper":
            self.ddpg.save_model_paper()
        else:
            self.ddpg.save_model()

        return self.env.get_hit_rate()
    
    def online_learning(self):
        if self.model == "paper":
            self.ddpg.load_model_paper()
        else:
            self.ddpg.load_model()

        total_rewards, done = 0.0, False
        #state is flatten and state shape is (3C,)
        current_state, done = self.env.reset()
        while(not done):
                
            proto_actor = self.ddpg.actor.get_proto_actor(current_state)

            knn_action_space = self.knn.expand_by_KNN(proto_actor)

            index, predict_reward = self.ddpg.critic.get_best_q_value_and_action_index(current_state, knn_action_space)
            best_action = int(knn_action_space[index])

            next_state, reward, done = self.env.step(best_action)

            #reward_error = predict_reward - reward
            #self.predict_rewards.append(predict_reward)
            #self.actual_rewards.append(reward)
            #print("DEBUG ACUTION REWARD-ERROR HIT-RATE:",best_action, reward_error, self.env.get_hit_rate())

            self.ddpg.brain.remember(current_state, best_action, reward, next_state, done)
            self.ddpg.replay()

            current_state = next_state
            total_rewards += reward

        if self.model == "paper":
            self.ddpg.save_model_paper()
        else:
            self.ddpg.save_model()
            
        cur_hit_rate = self.env.get_hit_rate()
        print("total rewards:"+str(total_rewards)+" hit rate:"+str(cur_hit_rate))
        if self.model == "paper":
            f = open("result/paper_hit_history.txt","w")
            f.write(str(self.env.get_hit_history())+"\n")
            f.close()
        else:
            f = open("result/new_hit_history.txt","w")
            f.write(str(self.env.get_hit_history())+"\n")
            f.close()
        f2 = open("result/pre_reward.txt","w")
        f2.write(str(self.predict_rewards)+"\n")
        f2.close()
        f3 = open("result/act_reward.txt","w")
        f3.write(str(self.actual_rewards)+"\n")
        f3.close()

        return cur_hit_rate

    def clean(self):
        del self.env
        del self.KNN_fraction
        del self.cache_size
        del self.K
        del self.state_shape
        del self.DEBUG
        del self.predict_rewards
        del self.actual_rewards
        self.ddpg.clean()
        self.knn.clean()
        del self.ddpg
        del self.knn
        gc.collect()
from ddpg import ddpg
from knn import knn
import os.path

#this class implements wolpertinger architecture
class wolpertinger:
    def __init__(self, cache_env, cache_size, model="new", KNN_fraction = 0.2, gamma = 0.9, tau=0.001):
        self.env = cache_env
        self.KNN_fraction = KNN_fraction
        self.cache_size = cache_size
        self.K = round((self.cache_size+1) * self.KNN_fraction)
        self.state_shape = self.env.get_state_shape()
        self.predict_rewards=[]
        self.actual_rewards=[]
        self.tau = tau
        self.model = model

        self.ddpg = ddpg(self.state_shape, self.cache_size, model, gamma, self.tau)
        self.knn = knn(self.cache_size, self.K)
    
    def offline_train(self, max_episodes):
        #check whether there are existed trained model
        if self.model == "paper":
            if(os.path.isfile("offline_model/actor_paper.h5")):
                self.ddpg.load_model_paper()
        else:
            if(os.path.isfile("offline_model/actor.h5")):
                self.ddpg.load_model()

        episodes, total_rewards, done = 0, 0.0, False
        #state is flatten and state shape is (3C,)
        current_state, done = self.env.reset()
        #start training
        while(episodes < max_episodes):
            if done:
                episodes +=1
                cur_hit_rate = self.env.get_hit_rate()
                print("episode:"+str(episodes)+" total rewards:"+str(total_rewards)+" hit rate:"+str(cur_hit_rate))

                self.env.hit_history.clear()
                total_rewards, done = 0.0, False
                current_state, done = self.env.reset()
                continue
            #get the proto actor
            proto_actor = self.ddpg.actor.get_proto_actor(current_state)
            #expand to knn action space
            knn_action_space = self.knn.expand_by_KNN(proto_actor)
            #find the actino with highest q-value
            index, _ = self.ddpg.critic.get_best_q_value_and_action_index(current_state, knn_action_space)
            best_action = int(knn_action_space[index])
            #apple the action to environment and receive reward and next state
            next_state, reward, done = self.env.step(best_action)
            #store transition
            self.ddpg.brain.remember(current_state, best_action, reward, next_state, done)
            #ddpg replay learning
            self.ddpg.replay()

            current_state = next_state
            total_rewards += reward

        if self.model == "paper":
            self.ddpg.save_model_paper()
        else:
            self.ddpg.save_model()
    
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

            index, _ = self.ddpg.critic.get_best_q_value_and_action_index(current_state, knn_action_space)
            best_action = int(knn_action_space[index])

            next_state, reward, done = self.env.step(best_action)

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
        return cur_hit_rate
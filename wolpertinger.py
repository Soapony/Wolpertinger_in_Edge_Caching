from ddpg import ddpg
from knn import knn
import os.path

#this class implements wolpertinger architecture
class wolpertinger:
    def __init__(self, cache_env, cache_size, model="proposed", KNN_fraction = 0.2, gamma = 0.9, tau=0.001):
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
    
    #offline phase
    def offline_train(self, max_episodes):
        #check whether there are existed trained model
        if self.model == "original":
            if(os.path.isfile("offline_model/actor_original.h5")):
                self.ddpg.load_model_original()
        else:
            if(os.path.isfile("offline_model/actor.h5")):
                self.ddpg.load_model_proposed()

        episodes, done = 0, False
        #state is flatten and state shape is (3M,)
        current_state, done = self.env.reset()
        #start training
        while(episodes < max_episodes):
            if done:
                episodes +=1
                self.env.hit_history.clear()
                done = False
                current_state, done = self.env.reset()
                continue
            #get the proto action
            proto_action = self.ddpg.actor.get_proto_action(current_state)
            #expand to a action set with k actions by KNN
            knn_action_space = self.knn.expand_by_KNN(proto_action)
            #find the action with the highest q-value
            index, _ = self.ddpg.critic.get_best_q_value_and_action_index(current_state, knn_action_space)
            best_action = int(knn_action_space[index])
            #apply the action to environment and receive reward and next state
            next_state, reward, done = self.env.step(best_action)
            #store transition
            self.ddpg.brain.remember(current_state, best_action, reward, next_state, done)
            #ddpg replay
            self.ddpg.replay()
            #move to next state
            current_state = next_state

        #after offline phase, save the parameters
        if self.model == "original":
            self.ddpg.save_model_original()
        else:
            self.ddpg.save_model_proposed()
    
    def online_learning(self):
        #load parameters
        if self.model == "original":
            self.ddpg.load_model_original()
        else:
            self.ddpg.load_model_proposed()

        done = False
        #state is flatten and state shape is (3C,)
        current_state, done = self.env.reset()
        while(not done):
            #get the proto action
            proto_actor = self.ddpg.actor.get_proto_action(current_state)
            #expand to a action set with k actions by KNN
            knn_action_space = self.knn.expand_by_KNN(proto_actor)
            #find the action with the highest q-value
            index, _ = self.ddpg.critic.get_best_q_value_and_action_index(current_state, knn_action_space)
            best_action = int(knn_action_space[index])
            #apply the action to environment and receive reward and next state
            next_state, reward, done = self.env.step(best_action)
            #store transition
            self.ddpg.brain.remember(current_state, best_action, reward, next_state, done)
            self.ddpg.replay()
            #move to next state
            current_state = next_state
        
        #save the learned parameters from the online phase
        if self.model == "original":
            self.ddpg.save_model_original()
        else:
            self.ddpg.save_model_proposed()
        
        cur_hit_rate = self.env.get_hit_rate()
        print("hit rate:"+str(cur_hit_rate))
        #save the cache hit rates during the whole online phase
        if self.model == "original":
            f = open("result/original_hit_history.txt","w")
            f.write(str(self.env.get_hit_history())+"\n")
            f.close()
        else:
            f = open("result/proposed_hit_history.txt","w")
            f.write(str(self.env.get_hit_history())+"\n")
            f.close()
        return cur_hit_rate
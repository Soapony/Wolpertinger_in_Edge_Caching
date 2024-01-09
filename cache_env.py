import numpy as np
import gc

class cache_env():
    def __init__(self, cache_size, requests_list, model="new", DEBUG=False, reward_dis_fac=0.9, online=False):
        #short- medium- long-term
        self.terms=np.array([10,100,1000])
        #self.terms=np.array([10,50,100,250,500,750,1000])
        #a list include all the requests
        self.all_requests = requests_list
        self.cache_size = cache_size
        self.state_shape = (len(self.terms)*(cache_size+1),)
        self.reward_discount_factor = reward_dis_fac
        self.DEBUG = DEBUG
        self.model = model
        self.online = online
        #feature space dict structure:
        #{unique content id: [short f, medium f, long f],
        # unique content id: [short f, medium f, long f],
        # ...}
        self.feature_space = {}
        #current request index in all_requests
        self.cur_req_ind = 0
        self.cache_hit_rate = 0.
        self.hit_history = []
        self.hit_count = 0
        self.last_req_ind = len(self.all_requests)-1
        self.previous_reward = 0.

        if self.DEBUG:
            print("============================DEBUG===================================")
            print("cache_env initialization summry:")
            print("all requests = ",self.all_requests)
            print("cache size = ",self.cache_size)
            print("state shape = ",self.state_shape)
            print("rewards discount factor = ",self.reward_discount_factor)
    
    def reset(self):
        self.feature_space = {}
        self.cur_req_ind = 0
        self.hit_count = 0
        self.cache_hit_rate = 0.0
        next_state, done = self.get_next_state()
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In cache_env -> reset -> next_state:")
            print(next_state)
        return next_state, done
    
    def step(self, cache_index):
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In cache_env -> step -> beginning -> feature_space:")
            print(self.feature_space)

        if(cache_index != 0):
            #add the new cache content to the last
            self.feature_space[self.all_requests[self.cur_req_ind]] = [1] * len(self.terms)
            tuples = list(self.feature_space.items())
            tmp_ind = len(tuples)-1 #last tuple index / new cache content index
            #exchange the position with the new content and removing cache
            tuples[cache_index-1], tuples[tmp_ind] = tuples[tmp_ind], tuples[cache_index-1]
            #then cut the last tuple which is the removing one and transform into a new dict
            self.feature_space = dict(tuples[:tmp_ind])
        self.update_feature_space()
        self.cache_hit_rate = self.hit_count / (self.cur_req_ind+1)
        if self.online:
            self.hit_history.append(self.cache_hit_rate)
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In cache_env -> step -> new feature_space:")
            print(self.feature_space)
            print("cache hit rate = ",self.cache_hit_rate)
        
        tmp_reward = 0.
        done = False
        next_state = []
        if(self.cur_req_ind == self.last_req_ind):
            if self.DEBUG:
                print("============================DEBUG===================================")
                print("In cache_env -> step -> cur_req_ind = last_req_ind")
            #reach the last state, no more next state
            done = True
            #To avoid network shape error and the batch size dismatch, get a meaningless next_state
            next_state = self.get_state_space()
            #no more requests so reward = 0
        else:
            tmp_reward = self.get_reward()
            self.cur_req_ind += 1
            next_state, done = self.get_next_state()

        reward = 0.
        if self.model == "paper":
            reward = tmp_reward
        else:
            reward = tmp_reward - self.previous_reward
            self.previous_reward = tmp_reward

        return next_state, reward, done
    
    #reward = next req hit count[0,1] + discount factor * next 100 req hit count[0,100]
    def get_reward(self):
        reward = self.get_short_term_reward()
        reward += self.reward_discount_factor * self.get_long_term_reward()
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In cache_env -> get_reward -> reward = ",reward)
        return reward

    def get_short_term_reward(self):
        next_req = self.all_requests[self.cur_req_ind+1]
        if next_req in self.feature_space:
            return 1.0
        return 0.0
    
    def get_long_term_reward(self):
        tmp_ind = self.cur_req_ind+1
        hit_count = 0
        i=0
        while(tmp_ind <= self.last_req_ind and i<100):
            if self.all_requests[tmp_ind] in self.feature_space:
                hit_count += 1
            i+=1
            tmp_ind += 1
        return hit_count
    
    def get_state_space(self):
        #transform feature space dict into state space format
        flatten_feature_space = np.concatenate(list(self.feature_space.values()))
        #add current request feature to the front
        state_space = np.insert(flatten_feature_space,0,[0]*len(self.terms))
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In cache_env -> get_state_space -> state_space:")
            print(state_space)
        return state_space
    
    def get_next_state(self):
        self.run_until_miss_and_cache_full()
        done = False
        if(self.cur_req_ind == self.last_req_ind+1):
            if self.DEBUG:
                print("============================DEBUG===================================")
                print("In cache_env -> get_next_state -> cur_req_ind = last_req_ind+1")
            #hit all remaining requests, done
            #no more next state, get a latest feature state to avoid Error
            done = True
        #still has next miss/state in request list
        return self.get_state_space(), done
    
    def run_until_miss_and_cache_full(self):
        while self.cur_req_ind <= self.last_req_ind:
            content_id = self.all_requests[self.cur_req_ind]
            if(self.hit_or_not_full(content_id)):
                continue
            break
    
    def hit_or_not_full(self, id):
        if id in self.feature_space:
            #update feature space
            for i in range(len(self.terms)):
                self.feature_space[id][i] += 1
            self.hit_count += 1
        elif(len(self.feature_space) < self.cache_size):
            #add to cache and update feature space
            self.feature_space[id] = [1] * len(self.terms)
        else:
            return False
        
        self.cache_hit_rate = self.hit_count / (self.cur_req_ind+1)
        if self.online:
            self.hit_history.append(self.cache_hit_rate)
            
        self.update_feature_space()
        self.cur_req_ind += 1
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In cache_env -> hit_or_not_full -> feature_space:")
            print(self.feature_space)
            print("cache hit rate = ",self.cache_hit_rate)
        
        return True
    
    def update_feature_space(self):
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In cache_env -> update_feature_space")
        indexes = self.cur_req_ind - self.terms
        #get short/medium/long term id(the previous 10/100/1000th request) 
        #and decrease by 1 if id is in feature space
        #because that id is going to out of terms range
        for i in range(len(indexes)):
            if(indexes[i] >= 0):
                content_id = self.all_requests[indexes[i]]
                if content_id in self.feature_space:
                    self.feature_space[content_id][i] = max(0, self.feature_space[content_id][i] - 1)
    
    def get_state_shape(self):
        return self.state_shape
    
    def get_hit_rate(self):
        return self.cache_hit_rate
    
    def get_hit_history(self):
        return self.hit_history
    
    def clean(self):
        del self.terms
        del self.all_requests
        del self.cache_hit_rate
        del self.cache_size
        del self.state_shape
        del self.reward_discount_factor
        del self.DEBUG
        del self.cur_req_ind
        del self.hit_count
        del self.last_req_ind
        del self.feature_space
        gc.collect()
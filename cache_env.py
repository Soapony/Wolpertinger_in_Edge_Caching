import numpy as np

#this class implements environment
class cache_env():
    def __init__(self, cache_size, requests_list, model="proposed", reward_dis_fac=0.9):
        #short- medium- long-term
        self.terms=np.array([10,100,1000])
        #a list include all the requests
        self.all_requests = requests_list
        self.cache_size = cache_size
        self.state_shape = (len(self.terms)*(cache_size+1),)
        self.reward_discount_factor = reward_dis_fac    #discount the long-term reward
        self.model = model
        #feature space dict structure:
        #{unique content id: [short f, medium f, long f],
        # unique content id: [short f, medium f, long f],
        # ...}
        self.feature_space = {}
        #current request index in all_requests
        self.cur_req_ind = 0
        self.cache_hit_rate = 0.
        self.hit_history = []   #record the cache hit rate during running
        self.hit_count = 0
        self.last_req_ind = len(self.all_requests)-1
        self.previous_reward = 0.
    
    #reset environment
    def reset(self):
        self.feature_space = {}
        self.cur_req_ind = 0
        self.hit_count = 0
        self.cache_hit_rate = 0.0
        next_state, done = self.get_next_state()
        return next_state, done
    
    #process the action into the environment and return the reward & next_state to agent
    def step(self, cache_index):
        if(cache_index != 0):
            #cache the new content according the index then update the feature space
            #add the new cache content to the last first
            self.feature_space[self.all_requests[self.cur_req_ind]] = [1] * len(self.terms)
            tuples = list(self.feature_space.items())
            tmp_ind = len(tuples)-1 #last tuple index which is newly cached content index
            #exchange the position with the newly cached content and the removed cache content
            tuples[cache_index-1], tuples[tmp_ind] = tuples[tmp_ind], tuples[cache_index-1]
            #then remove the last tuple which is the removed content and transform into a new dict structure
            self.feature_space = dict(tuples[:tmp_ind])
        #update the new feature space
        self.update_feature_space()
        self.cache_hit_rate = self.hit_count / (self.cur_req_ind+1)
        self.hit_history.append(self.cache_hit_rate)
        
        tmp_reward = 0.
        done = False
        next_state = []
        if(self.cur_req_ind == self.last_req_ind):
            #reach the last state, no more next state
            done = True
            #To avoid network shape error and the batch size dismatch, get a meaningless next_state
            next_state = self.get_state_space()
            #no more requests so reward remains 0
        else:
            #calculate combined rewards
            tmp_reward = self.get_combined_rewards()
            #move to t+1
            self.cur_req_ind += 1
            next_state, done = self.get_next_state()

        reward = 0.
        if self.model == "original":
            reward = tmp_reward         #original framework's reward
        else:
            reward = tmp_reward - self.previous_reward  #proposed reward mechnism
            self.previous_reward = tmp_reward           #record the combined rewards for next epoch

        return next_state, reward, done
    
    #combined reward = next req hit count[0,1] + discount factor * next 100 req hit count[0,100]
    def get_combined_rewards(self):
        reward = self.get_short_term_reward()
        reward += self.reward_discount_factor * self.get_long_term_reward()
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
        #avoid index out of range
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
        return np.insert(flatten_feature_space,0,[0]*len(self.terms))
    
    def get_next_state(self):
        self.run_until_miss_and_cache_full()
        done = False
        if(self.cur_req_ind == self.last_req_ind+1):
            #hit all remaining requests, done
            #no more next state, get a latest feature state to avoid Error
            done = True

        return self.get_state_space(), done
    
    #keep serving the requests which hit the cache or still have storage
    def run_until_miss_and_cache_full(self):
        while self.cur_req_ind <= self.last_req_ind:
            content_id = self.all_requests[self.cur_req_ind]
            if(self.hit_or_not_full(content_id)):
                continue
            break
    
    #if the incoming request hits the cache or still have storage
    def hit_or_not_full(self, id):
        #cache hit
        if id in self.feature_space:
            #update feature space
            for i in range(len(self.terms)):
                self.feature_space[id][i] += 1
            self.hit_count += 1
        #still have storage
        elif(len(self.feature_space) < self.cache_size):
            #add to cache and update feature space
            self.feature_space[id] = [1] * len(self.terms)
        #miss and storage full
        else:
            return False
        
        #update and record cache hit rate, update feature space
        self.cache_hit_rate = self.hit_count / (self.cur_req_ind+1)
        self.hit_history.append(self.cache_hit_rate)
        self.update_feature_space()
        #move to t+1
        self.cur_req_ind += 1
        
        return True
    
    def update_feature_space(self):
        #get the indexes of the previous 10th, 100th, and 1000th request
        indexes = self.cur_req_ind - self.terms
        #for the three indexes
        for i in range(len(indexes)):
            if(indexes[i] >= 0):    #avoid negative indexes
                content_id = self.all_requests[indexes[i]]  #get the previous requested content id
                #decrease the frequencies by 1 if the id is in feature space
                if content_id in self.feature_space:
                    self.feature_space[content_id][i] = max(0, self.feature_space[content_id][i] - 1)
    
    def get_state_shape(self):
        return self.state_shape
    
    def get_hit_rate(self):
        return self.cache_hit_rate
    
    def get_hit_history(self):
        return self.hit_history
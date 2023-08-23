from collections import deque
import random
import numpy as np
import gc

#memory class act as brain in DDPG, to remember action,state,reward and can memorize those
class memory:
    def __init__(self, DEBUG=False, batch_size=100, memory_size=10000):
        self.memory = deque(maxlen=memory_size)
        self.max_size = memory_size
        self.batch_size = batch_size
        self.DEBUG = DEBUG
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("Memory initialization summry:")
            print("max_size = ",self.max_size)
            print("memory queue = ",self.memory)
            print("batch size = ",batch_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state,action,reward,next_state,done])
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In memory -> remember")
            print("memory queue status:")
            print(self.memory)
    
    def memorize(self):
        samples = random.sample(self.memory, self.batch_size)
        samples = np.array(samples).T
        states, actions, rewards, next_states, dones = [np.vstack(samples[i, :]).astype(np.float) for i in range(5)]
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In memory -> memorize")
            print("===================samples===================")
            print(samples)
            print("===================states===================")
            print(states)
            print("===================actions===================")
            print(actions)
            print("===================rewawrds===================")
            print(rewards)
            print("===================next states===================")
            print(next_states)
            print("===================dones===================")
            print(dones)
        return states, actions, rewards, next_states, dones
    
    def clean(self):
        del self.memory
        del self.max_size
        del self.batch_size
        del self.DEBUG
        gc.collect()
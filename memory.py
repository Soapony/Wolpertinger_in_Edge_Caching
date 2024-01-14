from collections import deque
import random
import numpy as np

#memory class act as brain in DDPG, to remember action,state,reward and can memorize those
class memory:
    def __init__(self, batch_size=100, memory_size=10000):
        self.memory = deque(maxlen=memory_size)
        self.max_size = memory_size
        self.batch_size = batch_size

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state,action,reward,next_state,done])
    
    def memorize(self):
        samples = random.sample(self.memory, self.batch_size)
        samples = np.array(samples).T
        states, actions, rewards, next_states, dones = [np.vstack(samples[i, :]).astype(np.float) for i in range(5)]
        return states, actions, rewards, next_states, dones
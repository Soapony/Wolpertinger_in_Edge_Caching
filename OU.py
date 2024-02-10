import numpy as np
import random
import copy

#this class implements noise function
class OrnsteinUhlenbeckProcess(object):
    def __init__(self, size=1, mu=0., theta=0.3, sigma=0.9):
        self.mu = mu*np.ones(size)
        self.sigma = sigma
        self.theta = theta
        self.size = size

    #reset the state variable
    def reset(self):
        self.state = copy.copy(self.mu)

    #generate noise
    def generate(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return float(self.state)
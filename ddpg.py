import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from critic_network import critic_network
from actor_network import actor_network
from memory import memory
import numpy as np

tf.keras.backend.set_floatx('float64')

#this class implements ddpg functionality which create actor-critic netowrk and also target network and replay buffer
class ddpg:
    def __init__(self, state_shape, cache_size, framework="proposed", gamma=0.9, tau=0.001):

        self.actor = actor_network(state_shape, cache_size, framework)
        self.actor_t = actor_network(state_shape, cache_size, framework)
        self.critic = critic_network(state_shape, framework)
        self.critic_t = critic_network(state_shape, framework)
        self.brain = memory()
        self.gamma = gamma
        self.cache_size = cache_size
        self.tau = tau
        self.framework = framework
        #initialize the target networks' parameters aligned with actor&critic network
        self.update_target_para(self.actor.model, self.actor_t.model, tau=1.)
        self.update_target_para(self.critic.model, self.critic_t.model, tau=1.)

    #To update target network parameters slowly
    def update_target_para(self, model, target_model, tau=0.01):
        paras = model.get_weights()
        target_paras = target_model.get_weights()
        for i in range(len(target_paras)):
            target_paras[i] = paras[i] * tau + target_paras[i] * (1. - tau)
        target_model.set_weights(target_paras)
    
    def save_model_proposed(self):
        self.actor.model.save_weights('offline_model/actor.h5')
        self.critic.model.save_weights('offline_model/critic.h5')
    
    def save_model_original(self):
        self.actor.model.save_weights('offline_model/actor_original.h5')
        self.critic.model.save_weights('offline_model/critic_original.h5')
    
    def load_model_proposed(self):
        self.actor.model.load_weights('offline_model/actor.h5')
        self.actor_t.model.load_weights('offline_model/actor.h5')
        self.critic.model.load_weights('offline_model/critic.h5')
        self.critic.model.load_weights('offline_model/critic.h5')

    
    def load_model_original(self):
        self.actor.model.load_weights('offline_model/actor_original.h5')
        self.actor_t.model.load_weights('offline_model/actor_original.h5')
        self.critic.model.load_weights('offline_model/critic_original.h5')
        self.critic.model.load_weights('offline_model/critic_original.h5')

    
    #DDPG replay function, memorize from brain(replay buffer), then calculate the loss and gradient, then update parameters
    def replay(self):
        if(len(self.brain.memory) < self.brain.batch_size):
            return
        states, actions, rewards, next_states, dones = self.brain.memorize()

        #calculate target values
        next_actions = np.rint(self.actor_t.model.predict(next_states))
        next_actions = np.clip(next_actions, 0, self.cache_size)  
        next_q_values = self.critic_t.model.predict([next_states, next_actions])
        target_q_values = rewards + next_q_values * self.gamma * (1. - dones)

        #calculate critic loss and update critic network
        with tf.GradientTape() as tape:
            q_values = self.critic.model([states, actions])
            critic_loss = None
            if self.framework == "original":
                critic_loss = tf.keras.losses.MSE(target_q_values, q_values)
            else:
                h = tf.keras.losses.Huber()
                critic_loss = h(target_q_values, q_values)

        critic_gradient = tape.gradient(critic_loss, self.critic.model.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_gradient, self.critic.model.trainable_variables))

        #calculate actor gradient and update actor network
        with tf.GradientTape() as tape:
            actions = self.actor.model(states)
            actor_loss = -tf.math.reduce_mean(self.critic.model([states, actions]))

        actor_gradient = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.model.trainable_variables))

        #update target network
        self.update_target_para(self.actor.model, self.actor_t.model, self.tau)
        self.update_target_para(self.critic.model, self.critic_t.model, self.tau)
        #reduce exploration
        self.actor.epsilon_decrease()
    
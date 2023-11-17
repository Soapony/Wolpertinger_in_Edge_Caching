import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from critic_network import critic_network
from actor_network import actor_network
from memory import memory
import numpy as np
import gc

tf.keras.backend.set_floatx('float64')

#this class implements ddpg functionality which create actor-critic netowrk and also target network and brain for replay
class ddpg:
    def __init__(self, state_shape, cache_size, model="new", DEBUG=False, gamma=0.9, tau=0.001):

        self.actor = actor_network(state_shape, cache_size, model, DEBUG)
        self.actor_t = actor_network(state_shape, cache_size, model, DEBUG)
        self.critic = critic_network(state_shape, model, DEBUG)
        self.critic_t = critic_network(state_shape, model, DEBUG)
        self.brain = memory(DEBUG)
        self.gamma = gamma
        self.cache_size = cache_size
        self.DEBUG = DEBUG
        self.tau = tau

        self.update_target_para(self.actor.model, self.actor_t.model, tau=1.)
        self.update_target_para(self.critic.model, self.critic_t.model, tau=1.)

        if self.DEBUG:
            print("============================DEBUG===================================")
            print("DDPG initialization summry:")
            print("gamma = ",self.gamma)
            print("cache size = ",self.cache_size)

    #To update target network parameters
    def update_target_para(self, model, target_model, tau=0.01):
        paras = model.get_weights()
        target_paras = target_model.get_weights()
        for i in range(len(target_paras)):
            target_paras[i] = paras[i] * tau + target_paras[i] * (1. - tau)
        target_model.set_weights(target_paras)
    
    def save_model(self):
        self.actor.model.save_weights('offline_model/actor.h5')
        self.critic.model.save_weights('offline_model/critic.h5')
    
    def save_model_paper(self):
        self.actor.model.save_weights('offline_model/actor_paper.h5')
        self.critic.model.save_weights('offline_model/critic_paper.h5')
    
    def load_model(self):
        self.actor.model.load_weights('offline_model/actor.h5')
        self.actor_t.model.load_weights('offline_model/actor.h5')
        self.critic.model.load_weights('offline_model/critic.h5')
        self.critic.model.load_weights('offline_model/critic.h5')

    
    def load_model_paper(self):
        self.actor.model.load_weights('offline_model/actor_paper.h5')
        self.actor_t.model.load_weights('offline_model/actor_paper.h5')
        self.critic.model.load_weights('offline_model/critic_paper.h5')
        self.critic.model.load_weights('offline_model/critic_paper.h5')

    
    #DDPG replay function, memorize from brain, then predict and evaluate again and update parameter
    def replay(self):
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In ddpg -> replay -> beginning")
            print("brain size = ",len(self.brain.memory))
            print("brain batch size = ",self.brain.batch_size)

        if(len(self.brain.memory) < self.brain.batch_size):
            return
        states, actions, rewards, next_states, dones = self.brain.memorize()

        next_actions = np.rint(self.actor_t.model.predict(next_states))
        next_actions = np.clip(next_actions, 0, self.cache_size)
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In ddpg -> replay -> next_actions:")
            print(next_actions)
        
        next_q_values = self.critic_t.model.predict([next_states, next_actions])
        target_q_values = rewards + next_q_values * self.gamma * (1. - dones)
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In ddpg -> replay -> next_q_values:")
            print(next_q_values)
            print("In ddpg -> replay -> target_q_values:")
            print(target_q_values)

        with tf.GradientTape() as tape:
            q_values = self.critic.model([states, actions])
            #TD_error = target_q_values - q_values
            #critic_loss = tf.math.reduce_mean(tf.math.square(TD_error))
            #critic_loss = tf.math.reduce_mean(tf.math.abs(TD_error))
            critic_loss = tf.keras.losses.MSE(target_q_values, q_values)
            #h = tf.keras.losses.Huber()
            #critic_loss = h(target_q_values, q_values)
            if self.DEBUG:
                print("============================DEBUG===================================")
                print("In ddpg -> replay -> critic network loss calculation")
                print("q_values:")
                print(q_values)
                print("TD_error:")
                #print(TD_error)
                print("critic_loss:")
                print(critic_loss)
        critic_gradient = tape.gradient(critic_loss, self.critic.model.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_gradient, self.critic.model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor.model(states)
            actor_loss = -tf.math.reduce_mean(self.critic.model([states, actions]))
            if self.DEBUG:
                print("============================DEBUG===================================")
                print("In ddpg -> replay -> actor network loss calculation")
                print("actions:")
                print(actions)
                print("actor_loss:")
                print(actor_loss)
        actor_gradient = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.model.trainable_variables))
        self.update_target_para(self.actor.model, self.actor_t.model, self.tau)
        self.update_target_para(self.critic.model, self.critic_t.model, self.tau)
        self.actor.epsilon_decrease()

    def clean(self):
        self.actor.clean()
        self.actor_t.clean()
        self.critic.clean()
        self.critic_t.clean()
        self.brain.clean()
        del self.actor_t
        del self.actor
        del self.critic
        del self.critic_t
        del self.gamma
        del self.brain
        del self.cache_size
        del self.DEBUG
        gc.collect()
    
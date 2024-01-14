import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from OU import OrnsteinUhlenbeckProcess

tf.keras.backend.set_floatx('float64')

class actor_network:
    def __init__(
            self,
            state_shape,
            cache_size,
            framework="new",
            lr=1e-4
    ):
        self.learning_rate = lr
        self.state_shape = state_shape
        self.l_units32 = 32
        self.l_units64 = 64
        self.l_units128 = 128
        self.l_units256 = 256
        self.l_units512 = 512
        self.l_units1024 = 1024
        self.l_units2048 = 2048
        self.cache_size = cache_size
        self.noise = OrnsteinUhlenbeckProcess()
        self.noise.reset()
        self.epsilon = 1.0
        self.min_epsilon = 0.1
        self.epsilon_decay = 1e-6
        
        if framework == "paper":
            self.model = self.create_paper_actor_network()
        else:
            self.model = self.create_actor_network()
        self.optimizer = Adam(learning_rate=lr)

    #create actor network by using tensorFlow keras   
    def create_actor_network(self):
        output_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        input_state = Input(shape=self.state_shape)
        L1 = Dense(self.l_units2048, name="Actor_L1", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(input_state)
        L2 = Dense(self.l_units1024, name="Actor_L2", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L1)
        L3 = Dense(self.l_units512, name="Actor_L3", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L2)
        L4 = Dense(self.l_units256, name="Actor_L4", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L3)
        L5 = Dense(self.l_units128, name="Actor_L5", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L4)
        L6 = Dense(self.l_units64, name="Actor_L6", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L5)
        output = Dense(1, name="Actor_Out", activation='tanh', kernel_initializer=output_init)(L6)
        model = Model(inputs=input_state, outputs=output)
        return model

    def create_paper_actor_network(self):
        output_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        input_state = Input(shape=self.state_shape)
        L1 = Dense(self.l_units256, name="Actor_L1", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(input_state)
        L2 = Dense(self.l_units128, name="Actor_L2", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L1)
        output = Dense(1, name="Actor_Out", activation='tanh', kernel_initializer=output_init)(L2)
        model = Model(inputs=input_state, outputs=output)
        return model
    
    #input the state into network and predict which action to take, action is an interger, then refine the action within cache size range
    def get_proto_actor(self, state):
        state = state.reshape((1,len(state)))
        tmp_act = float(np.absolute(self.model.predict(state) - self.noise.generate() * self.epsilon))
        action = round(tmp_act * self.cache_size)
        self.noise.reset()

        return np.clip(action, 0, self.cache_size)
    
    def epsilon_decrease(self):
        if self.epsilon - self.epsilon_decay > self.min_epsilon:
            self.epsilon -= self.epsilon_decay
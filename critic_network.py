import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np
import gc

tf.keras.backend.set_floatx('float64')

class critic_network:
    def __init__(
            self,
            state_shape,
            framework="new",
            DEBUG=False,
            lr=1e-3
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
        self.DEBUG = DEBUG

        if framework == "paper":
            self.model = self.create_paper_critic_network()
        else:
            self.model = self.create_critic_network()
        self.optimizer = Adam(learning_rate=lr)
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("Critic network initialization summry:")
            print("learning rate = ",self.learning_rate)
            print("state_shape = ",self.state_shape)
            print("model summary:")
            print(self.model.summary())
    
    #create critic network by using tensorFlow keras
    def create_critic_network(self):
        input_state = Input(shape=[self.state_shape[0]])
        input_action = Input(shape=[1])

        L1_state = Dense(self.l_units2048, name="critic_state_L1", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(input_state)
        L2_state = Dense(self.l_units1024, name="critic_state_L2", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L1_state)
        L3_state = Dense(self.l_units512, name="critic_state_L3", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L2_state)
        L4_state = Dense(self.l_units256, name="critic_state_L4", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L3_state)
        
        L1_action = Dense(self.l_units256, name="critic_action_L1")(input_action)

        concat = Concatenate()([L4_state, L1_action])
        L1_concat = Dense(self.l_units256, name="critic_concat_L1", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(concat)
        L2_concat = Dense(self.l_units128, name="critic_concat_L2", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L1_concat)
        L3_concat = Dense(self.l_units64, name="critic_concat_L3", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L2_concat)
        L4_concat = Dense(self.l_units32, name="critic_concat_L4", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L3_concat)

        output = Dense(1, name="critic_out", activation='linear')(L4_concat)
        model = Model(inputs=[input_state, input_action], outputs=output)

        return model
    
    def create_paper_critic_network(self):
        input_state = Input(shape=[self.state_shape[0]])
        input_action = Input(shape=[1])

        L1_state = Dense(self.l_units64, name="critic_state_L1", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(input_state)
        L2_state = Dense(self.l_units32, name="critic_state_L2", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L1_state)
        
        L1_action = Dense(self.l_units32, name="critic_action_L1")(input_action)

        concat = Concatenate()([L2_state, L1_action])
        L1_concat = Dense(self.l_units32, name="critic_concat_L1", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(concat)
        L2_concat = Dense(self.l_units32, name="critic_concat_L2", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L1_concat)

        output = Dense(1, name="critic_out", activation='linear')(L1_concat)
        model = Model(inputs=[input_state, input_action], outputs=output)

        return model

    #copy state for each action in action space, and input the actions and states as batches into network
    #and predict the values, then get the max value index which also is the action index and return
    def get_best_q_value_and_action_index(self, state, action_space):
        state=np.tile(state,(len(action_space),1))
        q_values = self.model.predict([state, action_space])
        index = np.argmax(q_values)
        q_value = q_values[index][0]
        if self.DEBUG:
            print("============================DEBUG===================================")
            print("In critic network -> get_best_q_value_and_action_index")
            print("q values = ",q_values)
            print("index = ",index)
            print("q value = ",q_value)
        return index, q_value
    
    def clean(self):
        del self.learning_rate
        del self.state_shape
        del self.l_units32
        del self.l_units64
        del self.l_units128
        del self.l_units256
        del self.l_units512
        del self.l_units1024
        del self.l_units2048
        del self.DEBUG
        del self.model
        del self.optimizer
        gc.collect()
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
import numpy as np

tf.keras.backend.set_floatx('float64')

#this class implements the critic network
class critic_network:
    def __init__(
            self,
            state_shape,
            framework="proposed",
            lr=1e-3
    ):
        #define hyper-parameters
        self.learning_rate = lr
        self.state_shape = state_shape
        self.l_units32 = 32
        self.l_units64 = 64
        self.l_units128 = 128
        self.l_units256 = 256
        self.l_units512 = 512
        self.l_units1024 = 1024
        self.l_units2048 = 2048

        #create critic network based on the seclected DRL agent
        if framework == "original" or True:
            self.model = self.create_original_critic_network()
        else:
            self.model = self.create_critic_network()
        self.optimizer = Adam(learning_rate=lr)
    
    #create critic network by using tensorFlow keras
    def create_critic_network(self):
        input_state = Input(shape=[self.state_shape[0]])
        input_action = Input(shape=[1])

        #state space go through 4 layers
        L1_state = Dense(self.l_units2048, name="critic_state_L1", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(input_state)
        L2_state = Dense(self.l_units1024, name="critic_state_L2", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L1_state)
        L3_state = Dense(self.l_units512, name="critic_state_L3", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L2_state)
        L4_state = Dense(self.l_units256, name="critic_state_L4", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L3_state)
        
        #action space go throug 1 layers
        L1_action = Dense(self.l_units256, name="critic_action_L1")(input_action)

        #combine two inputs together
        concat = Concatenate()([L4_state, L1_action])

        #combined imputs go through 4 layers
        L1_concat = Dense(self.l_units256, name="critic_concat_L1", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(concat)
        L2_concat = Dense(self.l_units128, name="critic_concat_L2", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L1_concat)
        L3_concat = Dense(self.l_units64, name="critic_concat_L3", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L2_concat)
        L4_concat = Dense(self.l_units32, name="critic_concat_L4", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L3_concat)

        output = Dense(1, name="critic_out", activation='linear')(L4_concat)
        model = Model(inputs=[input_state, input_action], outputs=output)

        return model
    
    def create_original_critic_network(self):
        input_state = Input(shape=[self.state_shape[0]])
        input_action = Input(shape=[1])

        #state space go through 2 layers
        L1_state = Dense(self.l_units64, name="critic_state_L1", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(input_state)
        L2_state = Dense(self.l_units32, name="critic_state_L2", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L1_state)
        
        #action space go through 1 layers
        L1_action = Dense(self.l_units32, name="critic_action_L1")(input_action)

        #combine two inputs together
        concat = Concatenate()([L2_state, L1_action])
        
        #combined inputs go through 2 layers
        L1_concat = Dense(self.l_units32, name="critic_concat_L1", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(concat)
        L2_concat = Dense(self.l_units32, name="critic_concat_L2", activation='relu', kernel_initializer=tf.keras.initializers.HeNormal())(L1_concat)

        output = Dense(1, name="critic_out", activation='linear')(L2_concat)
        model = Model(inputs=[input_state, input_action], outputs=output)

        return model

    #copy state for each action in action set, and input the actions and states as batches into network
    #and output the Q-values, then get the max Q-value index which also is the action index and return
    def get_best_q_value_and_action_index(self, state, action_space):
        state=np.tile(state,(len(action_space),1))
        q_values = self.model.predict([state, action_space])
        index = np.argmax(q_values)
        q_value = q_values[index][0]
        return index, q_value
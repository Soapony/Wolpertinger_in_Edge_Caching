import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
import keras.backend as K

K.set_learning_phase(1)

import tensorflow as tf



class Critic:
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

    def create_critic_model(self):
        state_input = Input(shape=[self.state_dim])
        state_h1 = Dense(64, activation='relu')(state_input)
        state_h2 = Dense(32, activation='relu')(state_h1)

        action_input = Input(shape=[self.action_dim])
        action_h1 = Dense(32)(action_input)

        merged = Concatenate()([state_h2, action_h1])
        merged_h1 = Dense(32, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model = Model(inputs=[state_input, action_input], outputs=output)

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return state_input, action_input, model

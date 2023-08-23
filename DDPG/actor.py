import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
import keras.backend as K

K.set_learning_phase(1)

import tensorflow as tf



class Actor:
    def __init__(self,
                 state_dim,
                 action_dim,
                 learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

    def create_actor_model(self):
        state_input = Input(shape=[self.state_dim])
        h1 = Dense(256, activation='relu')(state_input)
        h2 = Dense(128, activation='relu')(h1)

        output = Dense(self.action_dim, activation='tanh')(h2)

        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        return state_input, model

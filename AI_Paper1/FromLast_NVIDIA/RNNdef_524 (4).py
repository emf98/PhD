##This python file should contain the requisite import statements and definitions for the LSTM Architectures/GRU.
#Relevant to May of 2024. 

##start off with import statements ... These probably aren't necessary here, but I am including them anyway because better safe than sorry. :^)

##so-called "math" related imports
import numpy as np

##last ... ML imports
##import tensorflow/keras related files
import tensorflow as tf    

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, Activation, Reshape, Flatten, LSTM, Dense, Dropout, Embedding, Bidirectional, GRU
from tensorflow.keras import Sequential
from tensorflow.keras import initializers, regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import constraints
from tensorflow.keras.layers import Layer, InputSpec

######################################################
###LSTM Model]

#ntimestep = length of time series
#nfeature = number of features 
#numlayer = number of recurrent layers >=2
#regval = regularizers
#neurons = number of neurons in recurrent layers ... as list
#numdenselayer = number of dense layers
#denseneurons = number of neurons in dense layer
#out_neurons = number of categories
#lr = learning rate
#recurr_dropout = dropout %

def build_lstm(ntimestep, nfeature, numlayer, regval, neurons, numdenselayer, denseneurons, out_neurons, lr, recurr_dropout):
    
    input_tensor = Input(shape=(ntimestep, nfeature))
    layer1 = layers.LSTM(neurons[0], return_sequences=True, recurrent_dropout=recurr_dropout, kernel_regularizer=regularizers.l2(regval[0]))(input_tensor)
    if numlayer >=2:
        print('layer ' + str(numlayer))
        for i in range(1, numlayer-1):
            layer1 = layers.LSTM(neurons[i], return_sequences=True, recurrent_dropout=recurr_dropout, kernel_regularizer=regularizers.l2(regval[i]))(layer1)

    layer1 = layers.LSTM(neurons[numlayer-1], return_sequences=False, recurrent_dropout=recurr_dropout, kernel_regularizer=regularizers.l2(regval[numlayer-1]))(layer1)
    
    if numdenselayer > 0:
        for i in range(numdenselayer):
            layer1 = layers.Dense(denseneurons[i], activation="relu")(layer1)
            
    output_tensor = layers.Dense(out_neurons, activation='softmax',)(layer1)

    model = Model(input_tensor, output_tensor)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    #decay_rate = lr / epochs
    #momentum = 0.9

    model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=[keras.metrics.categorical_accuracy],)
                            
    return model
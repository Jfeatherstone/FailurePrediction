import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow.keras.backend as kb

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops


"""
Projected Categorical Crossentopy loss function
"""

def pCCE(target, output, weightVector, axis=-1):
    
    # scale preds so that the class probas of each sample sum to 1
    output = output / math_ops.reduce_sum(output, -1, True)

    # Take projections into the weight vectors
    outputProjection = tf.tensordot(output, weightVector, axes=1)
    # Don't take the projection of the target, only the prediction
    #targetProjection = tf.tensordot(target, weightVector, axes=1)

    # Make sure there are no invalid values in the logarithm
    epsilon_ = constant_op.constant(kb.epsilon(), output.dtype.base_dtype)
    outputProjection = clip_ops.clip_by_value(outputProjection, epsilon_, 1. - epsilon_)

    return -math_ops.reduce_sum(target * math_ops.log(outputProjection), -1)

"""
Weights for use with pCCE
"""

def linearWeight(sampleLength):
    wVec = np.zeros([sampleLength, sampleLength])
    for i in range(sampleLength):
        wVec[i] = np.abs([k - i for k in range(sampleLength)])
        wVec[i] = np.max(wVec[i]) - wVec[i]
        wVec[i] /= np.sum(wVec[i])

    return wVec

def deltaWeight(sampleLength):
    wVec = np.zeros([sampleLength, sampleLength], dtype=np.float32)
    for i in range(sampleLength):
        wVec[i,i] = 1
        
    return wVec.astype(dtype=np.float32)

def lorentzian(x, w=.5, c=0):
    return w**2 / (w**2 + (2*x - 2*c)**2)

def lorentzianWeight(sampleLength, width=1):
    wVec = np.zeros([sampleLength, sampleLength], dtype=np.float32)
    iArr = np.arange(sampleLength)
    for i in range(sampleLength):
        wVec[i] = lorentzian(iArr, w=width, c=i)
        
    return wVec

def gaussian(x, w=.5, c=0):
    return np.exp(-(x - c)**2 / (2*w**2))

def gaussianWeight(sampleLength, width=1):
    wVec = np.zeros([sampleLength, sampleLength], dtype=np.float32)
    iArr = np.arange(sampleLength)
    for i in range(sampleLength):
        wVec[i] = gaussian(iArr, w=width, c=i)
        
    return wVec


# This sets up our net that will be used to train on various scores
def initializeNet():
    # None shape so we can input a variable length vector
    #inputs = keras.Input(shape=(None,))

    #lstm = layers.LSTM(2)(inputs)

    #model = keras.Model(inputs=inputs, outputs=lstm, name='lstm_net')
    # Just use the basic sequential model, since we don't need any crazy
    # layer setups
    model = keras.Sequential()

    # Add the input layer. None shape so that we can pass a variable length
    # array in
    model.add(layers.Input(shape=(None,1)))

    # Add two LSTM layers. No particular reason for 2, just seeing how it works
    # The return_sequences lets you chain together LSTM layers, otherwise it will
    # mess with the shape
    model.add(layers.LSTM(1, return_sequences=True))
    model.add(layers.LSTM(1, return_sequences=True))

    model.compile(optimizer=keras.optimizers.Adam(), loss='mse', metrics=['accuracy'])

    return model

    

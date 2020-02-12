from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers


fc_id = 54 # FC6 Layer Number
#fc_id = 58 # FC7 Layer Number

rank = 6
sr = 0.5
rr = 0.5

def normalize(X_train,X_test):
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test

def normalize_production(x):
    mean = 120.707
    std = 64.15
    return (x-mean)/(std+1e-7)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train, x_test = normalize(x_train, x_test)

json_file = open('cifar10vgg.json', 'r')
cifar10_model_json = json_file.read()
json_file.close()
cifar10_model = model_from_json(cifar10_model_json)
cifar10_model.load_weights("cifar10vgg.h5")
cifar10_model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

def sparse_SVD_wr(weights,U,S,V,keep,sr,rr):
    tU, tS, tV = U[:, 0:keep], S[0:keep], V[0:keep, :]

    # Input node selection
    iwm = np.sum(abs(weights),axis=1)
    imid = sorted(iwm)[int(weights.shape[0]*sr)]
    ipl = np.where(iwm<imid)[0]

    # Output node selection
    owm = np.sum(abs(weights),axis=0)
    omid = sorted(owm)[int(weights.shape[1]*sr)]
    opl = np.where(owm<omid)[0]

    # Masking the weights
    subrank = int(keep*rr)
    for ind in ipl:
        tU[ind,subrank:]=0

    for ind in opl:
        tV[subrank:,ind]=0

    return tU, tS, tV


cifar10_model.load_weights("cifar10vgg.h5")
cifar10_model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
fc1 = cifar10_model.layers[fc_id].get_weights()

weights = fc1

keep = rank

# Decomposition and Reconstruction
U, S, V = np.linalg.svd(weights[0], full_matrices=False)
tU, tS, tV = sparse_SVD_wr(weights[0],U,S,V,keep,sr,rr)

weights_t = np.matmul(np.matmul(tU, np.diag(tS)), tV)

# Loading weights for new model
weights[0] = weights_t
cifar10_model.layers[fc_id].set_weights(weights)

# Write the testing input and output variables
score = cifar10_model.evaluate(x_test, y_test, verbose=0)
sparcesvd_accuracy = score[1]
print('Params:',tU.size+tS.size+tV.size,', Acc:', truncsvd_accuracy)
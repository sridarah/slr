from __future__ import print_function
import keras
from keras.datasets import cifar100
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

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, 100)
y_test = keras.utils.to_categorical(y_test, 100)

x_train, x_test = normalize(x_train, x_test)


json_file = open('cifar100vgg.json', 'r')
cifar100_model_json = json_file.read()
json_file.close()
cifar100_model = model_from_json(cifar100_model_json)
cifar100_model.load_weights("cifar100vgg.h5")
cifar100_model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

keep = rank
fc1 = cifar100_model.layers[fc_id].get_weights()

# Decomposition and Reconstruction
U, S, V = np.linalg.svd(fc1[0], full_matrices=False)
tU, tS, tV = U[:, 0:keep], S[0:keep], V[0:keep, :]
fc1_t = np.matmul(np.matmul(tU, np.diag(tS)), tV)

# Loading weights for new model
fc1[0] = fc1_t
cifar100_model.layers[fc_id].set_weights(fc1)

# Write the testing input and output variables
score = cifar100_model.evaluate(x_test,y_test,batch_size=50,verbose=0)
truncsvd_accuracy = score[1]
print('Params:',tU.size+tS.size+tV.size,', Acc:', truncsvd_accuracy)
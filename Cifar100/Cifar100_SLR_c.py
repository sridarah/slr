from __future__ import print_function
import keras
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json, Model
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

cifar100_model.load_weights("cifar100vgg.h5")
cifar100_model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

fc1 = cifar100_model.layers[fc_id].get_weights()

weights = fc1[0]

ip_node = weights.shape[0]
op_node = weights.shape[1]

U, S, V = np.linalg.svd(weights, full_matrices=False)
tU, tS, tV = U[:, 0:keep], S[0:keep], V[0:keep, :]

fc1[0] = np.matmul(np.matmul(tU, np.diag(tS)), tV)
cifar100_model.layers[fc_id].set_weights(fc1)

score = cifar100_model.evaluate(x_train, y_train, verbose=0)
svd_cost = score[0]

subrank = int(keep*rr)

imc = np.zeros([ip_node])

for ind in range(0,ip_node):
	tempU = np.copy(tU)
	tempU[ind,subrank:]=0
	weights_t = fc1
	weights_t[0] = np.matmul(np.matmul(tempU, np.diag(tS)), tV)
	cifar100_model.layers[fc_id].set_weights(weights_t)
	score = cifar100_model.evaluate(x_train, y_train, verbose=0)
	svd_cost_t = score[0]
	imc[ind] = abs(svd_cost-svd_cost_t)
	cifar100_model.layers[fc_id].set_weights(fc1)
	
omc = np.zeros([op_node])

for ind in range(0,op_node):
	temptV = np.copy(tV)
	temptV[subrank:,ind]=0
	weights_t = fc1
	weights_t[0] = np.matmul(np.matmul(tU, np.diag(tS)), temptV)
	cifar100_model.layers[fc_id].set_weights(weights_t)
	score = cifar100_model.evaluate(x_train, y_train, verbose=0)
	svd_cost_t = score[0]
	omc[ind] = abs(svd_cost-svd_cost_t)
	cifar100_model.layers[fc_id].set_weights(fc1)
	
# Loading Model
cifar100_model.load_weights("cifar100vgg.h5")
cifar100_model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

# Loading weights of the model
fc1 = cifar100_model.layers[fc_id].get_weights()

weights = fc1
U, S, V = np.linalg.svd(weights[0], full_matrices=False)
tU, tS, tV = U[:, 0:keep], S[0:keep], V[0:keep, :]

# Input node selection
imid = sorted(imc)[int(ip_node*sr)]
ipl = np.where(imc<imid)[0]

# Output node selection
omid = sorted(omc)[int(op_node*sr)]
opl = np.where(omc<omid)[0]

# Masking the weights
subrank = int(keep*rr)
for ind in ipl:
	tU[ind,subrank:]=0

for ind in opl:
	tV[subrank:,ind]=0
	
weights_t = np.matmul(np.matmul(tU, np.diag(tS)), tV)

# Loading weights for new model
weights[0] = weights_t
cifar100_model.layers[fc_id].set_weights(weights)

# Write the testing input and output variables
score = cifar100_model.evaluate(x_test, y_test, verbose=0)
slr_accuracy = score[1]
print('Params:',tU.size+tS.size+tV.size,', Acc:', slr_accuracy)
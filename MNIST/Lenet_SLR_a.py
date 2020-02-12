from keras import backend
from keras import datasets
import keras
import numpy as np
from keras import models, layers
from keras.models import Model,Sequential, model_from_json
from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten
from keras.datasets import mnist
from keras.utils import np_utils

fc_id = 5 # FC Layer Number
rank = 6
sr = 0.5
rr = 0.5

# Load dataset as train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols = x_train.shape[1:]

if backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = np.pad(x_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
x_test = np.pad(x_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

# Set numeric type to float32 from uint8
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize value to [0, 1]
x_train /= 255
x_test /= 255

# Transform lables to one-hot encoding
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# Reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 32,32,1)
x_test = x_test.reshape(x_test.shape[0], 32,32,1) 

json_file = open('Lenet.json', 'r')
lenet_model_json = json_file.read()
json_file.close()
lenet_model = model_from_json(lenet_model_json)
lenet_model.load_weights("Lenet.h5")
lenet_model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

def sparse_SVD_ar(U,S,V,inp_act,out_act,keep,sr,rr):
    
    tU, tS, tV = U[:, 0:keep], S[0:keep], V[0:keep, :]

    # Input node selection
    iwm = np.sum(abs(inp_act),axis=0)
    imid = sorted(iwm)[int(U.shape[0]*sr)]
    ipl = np.where(iwm<imid)[0]

    # Output node selection
    owm = np.sum(abs(out_act),axis=0)
    omid = sorted(owm)[int(V.shape[1]*sr)]
    opl = np.where(owm<omid)[0]
    
    # Masking the weights
    subrank = int(keep*rr)
    for ind in ipl:
        tU[ind,subrank:]=0

    for ind in opl:
        tV[subrank:,ind]=0

    return tU, tS, tV

# Loading Model
lenet_model.load_weights("Lenet.h5")
lenet_model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
act_fc1 = Model(input=lenet_model.input, output=lenet_model.get_layer('flatten_2').output)
act_fc2 = Model(input=lenet_model.input, output=lenet_model.get_layer('dense_4').output)

# Flatten 
inp_act = act_fc1.predict(x_train)
out_act = act_fc2.predict(x_train)

keep = rank

fc1 = lenet_model.layers[fc_id].get_weights()

weights = np.copy(fc1)

# Decomposition and Reconstruction
U, S, V = np.linalg.svd(weights[0], full_matrices=False)
tU, tS, tV = sparse_SVD_ar(U,S,V,inp_act,out_act,keep,sr,rr)

weights_t = np.matmul(np.matmul(tU, np.diag(tS)), tV)

# Loading weights for new model
weights[0] = weights_t
lenet_model.layers[fc_id].set_weights(weights)

# Write the testing input and output variables
score = lenet_model.evaluate(x_test, y_test, verbose=0)
slr_accuracy = score[1]
print('Params:',tU.size+tS.size+tV.size,', Acc:', slr_accuracy)
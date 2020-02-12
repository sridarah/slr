import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential, model_from_json, Model
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.callbacks import ModelCheckpoint
import itertools
import tensorflow as tf

fc_id = 20 # FC6 Layer Number
#fc_id = 21 # FC7 Layer Number
rank = 6
sr = 0.5
rr = 0.5

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

Train_path = 'data/train'
Valid_path = 'data/valid'
Test_path = 'data/test1'

train_batches = ImageDataGenerator().flow_from_directory(Train_path, target_size=(224,224),batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(Valid_path, target_size=(224,224),  batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(Test_path, target_size=(224,224), batch_size=10)

json_file = open('VGG16_CnD.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
vgg16_model = model_from_json(loaded_model_json)
vgg16_model.load_weights("VGG16_CnD.h5")
vgg16_model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

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


# FC6 Layer Number

#act_fc1 = Model(input=vgg16_model.input, output=vgg16_model.get_layer('flatten').output)
#act_fc2 = Model(input=vgg16_model.input, output=vgg16_model.get_layer('fc1').output)

# FC7 Layer Number

act_fc1 = Model(input=vgg16_model.input, output=vgg16_model.get_layer('fc1').output)
act_fc2 = Model(input=vgg16_model.input, output=vgg16_model.get_layer('fc2').output)

# Flatten 
inp_act = act_fc1.predict_generator(train_batches,steps=200)
out_act = act_fc2.predict_generator(train_batches,steps=200)
	

keep = rank

vgg16_model.load_weights("VGG16_CnD.h5")
vgg16_model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])

fc1 = vgg16_model.layers[fc_id].get_weights()

weights = fc1

# Decomposition and Reconstruction
U, S, V = np.linalg.svd(weights[0], full_matrices=False)
tU, tS, tV = sparse_SVD_ar(U,S,V,inp_act,out_act,keep,sr,rr)

weights_t = np.matmul(np.matmul(tU, np.diag(tS)), tV)

# Loading weights for new model
weights[0] = weights_t
vgg16_model.layers[fc_id].set_weights(weights)

# Write the testing input and output variables
score = vgg16_model.evaluate_generator(valid_batches,steps=200)
slr_accuracy = score[1]
print('Params:',tU.size+tS.size+tV.size,', Acc:', slr_accuracy)
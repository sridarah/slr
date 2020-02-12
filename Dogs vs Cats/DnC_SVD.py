import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential, model_from_json
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

keep = rank

fc1 = vgg16_model.layers[fc_id].get_weights()

# Decomposition and Reconstruction
U, S, V = np.linalg.svd(fc1[0], full_matrices=False)
tU, tS, tV = U[:, 0:keep], S[0:keep], V[0:keep, :]
fc1_t = np.matmul(np.matmul(tU, np.diag(tS)), tV)

# Loading weights for new model
fc1[0] = fc1_t
vgg16_model.layers[fc_id].set_weights(fc1)

score = vgg16_model.evaluate_generator(valid_batches,steps=200)
truncsvd_accuracy = score[1]
print('Params:',tU.size+tS.size+tV.size,', Acc:', truncsvd_accuracy)
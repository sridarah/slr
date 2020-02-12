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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

Train_path = 'data/train'
Valid_path = 'data/valid'
Test_path = 'data/test1'

train_batches = ImageDataGenerator().flow_from_directory(Train_path, target_size=(224,224),batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(Valid_path, target_size=(224,224),  batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(Test_path, target_size=(224,224), batch_size=10)

# Uncomment the following line if you don't have the pretrained VGG16
#vgg16_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)

# Comment the following line if you don't have the pretrained VGG16
json_file = open('VGG16.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
vgg16_model = model_from_json(loaded_model_json)
vgg16_model.load_weights("VGG16.h5")
vgg16_model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
print("Loaded model from disk")

vgg16_model.summary()
vgg16_model.layers.pop()

model = Sequential()
for layer in vgg16_model.layers:
    model.add(layer)

model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=5, 
                    validation_data=valid_batches, validation_steps=5, epochs=50, verbose=2)

					
model_json = model.to_json()
with open("VGG16_CnD.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("VGG16_CnD.h5")
print("Saved model to disk")
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
Valid_path = 'data/test'
Test_path =  'data/valid'

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_batches = train_datagen.flow_from_directory(Train_path, target_size=(224,224),batch_size=64)
valid_batches = ImageDataGenerator().flow_from_directory(Valid_path, target_size=(224,224),  batch_size=64)
test_batches = ImageDataGenerator().flow_from_directory(Test_path, target_size=(224,224), batch_size=64)

# Uncomment the following line if you don't have the pretrained VGG19
#vgg19_model = keras.applications.vgg19.VGG19(weights='imagenet', include_top=True)

# Comment the following line if you don't have the pretrained VGG19
json_file = open('VGG19.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
vgg19_model = model_from_json(loaded_model_json)
vgg19_model.load_weights("VGG19.h5")
vgg19_model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
print("Loaded model from disk")

vgg19_model.layers.pop()

model = Sequential()
for layer in vgg19_model.layers:
    model.add(layer)

model.add(Dense(102, activation='softmax'))

model.summary()

model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch=128, 
                    validation_data=valid_batches, validation_steps=128, epochs=50, verbose=1)

model_json = model.to_json()
with open("VGG19_Oxf.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("VGG19_Oxf.h5")
print("Saved model to disk")
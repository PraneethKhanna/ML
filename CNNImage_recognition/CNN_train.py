

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import pickle
from keras.models import load_model


#initializing
classifier = Sequential()
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,3), activation='relu'))

#pooling

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#flattening
classifier.add(Flatten())

#full connection
classifier.add(Dense(units = 100, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory('C:/Users/pxb171530/Downloads/dataset/training_set',target_size=(64, 64),batch_size=32,class_mode='binary')

test = test_datagen.flow_from_directory('C:/Users/pxb171530/Downloads/dataset/test_set',target_size=(64, 64),batch_size=32,class_mode='binary')

classifier.fit_generator(
        train,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test,
        validation_steps=2000,use_multiprocessing=False)






classifier.save('C:/Users/pxb171530/Downloads/2layered_pickel.pkl') 

classifier.p
'''
import numpy as np
import cv2

train.class_indices

testimg=  cv2.imread('C:/Users/pxb171530/Downloads/dataset/single_prediction/who.jpg')
testimg=cv2.resize(testimg,(64,64))
testimg= np.expand_dims(testimg,axis=0)

result=classifier.predict(testimg)
'''






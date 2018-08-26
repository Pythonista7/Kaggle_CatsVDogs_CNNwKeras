#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 20:21:29 2018

@author: ashwin

Note: First time you run the code please check the following :
	* The TRAIN_IMG_DIR and TEST_IMG_DIR must have the paths of the testing and training images
	* Uncomment the pre_process_training _data to generate the npy file we use for training 
	* The first time you run pre process .npy is save and the un-commented line can again be commented for furtur use
"""

import numpy as np
import cv2 as img
import keras
from keras.models import Sequential
from keras.layers import  Flatten, Dense ,Convolution2D,MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
import os
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


TRAIN_IMG_DIR='/home/ashwin/Desktop/Datasets/CatsVDogs/train'
TEST_IMG_DIR='/home/ashwin/Desktop/Datasets/CatsVDogs/test1'

IMG_SIZE=100

#dataset=ImageDataGenerator()



def pre_process_training_imgs():

    train_list=os.listdir(TRAIN_IMG_DIR)
    training_data=[]
    X=[]
    y=[]
    for i in range(len(train_list)):
        img_data=img.resize(img.imread(TRAIN_IMG_DIR+'/'+train_list[i],0),(IMG_SIZE,IMG_SIZE))
        img_data=np.array(img_data)
        label=train_list[i].split('.')[0]
        training_data.append([img_data,label])
        if i % 5000==0: print("Loaded ",i)

        '''

        #One-hot-encode the labels
        if train_list[i].split('.')[0]=='dog':
            label=[1,0]
        else:
            label=[0,1]
        '''

    for data in training_data:
        X.append(data[0])
        if data[1] == 'cat':
            #[0,1] is CAT
            y.append([0,1])
        else:
            #[1,0] is DOG
            y.append([1,0])

    np.save('X_data.npy',np.array(X))
    np.save('y_data.npy',np.array(y))
    #training_data=[np.array(X),y]
    #np.save('training_data.npy',save_training_data)

#pre_process_training_imgs()

X=np.load('X_data.npy')
y=np.load('y_data.npy')
#training_data=np.load('save_training_data.npy')

#X_train,y_train,X_test,y_test

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,shuffle=True)#training_data[0],training_data[1],test_size=0.33,shuffle=True)
X_train=np.array(X_train)
X_train=X_train.reshape((16750,IMG_SIZE,IMG_SIZE,1))#X_train.reshape((16750,50,50,1))
X_test=np.array(X_test)
X_test=X_test.reshape((8250,IMG_SIZE,IMG_SIZE,1))

def conv_net():

    model=Sequential()
    print 'Sequential check'
    model.add(Convolution2D(32,(3,3),input_shape=(IMG_SIZE,IMG_SIZE,1),activation='relu'))
    print 'Conv2D layer 1'
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Convolution2D(50,(2,2),activation='relu'))
    print 'Conv2D layer 2'
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    
    model.add(Convolution2D(100,(2,2),activation='relu'))
    print 'Conv2D layer 3'
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    
    model.add(Flatten())

    #model.add(Dense(units=7*7*16750, activation='relu'))
    model.add(Dense(units=500, activation='relu'))
    
    model.add(Dense(100,activation='relu'))
    
    model.add(Dense(units=2, activation='softmax'))

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    return model



def training_step(X_train,y_train,X_test,y_test):

    model=conv_net()
    
    datagen = ImageDataGenerator()

    datagen.fit(X_train)

    model.fit_generator(datagen.flow(X_train, y_train, batch_size=25),
                    steps_per_epoch=len(X_train)/25 , epochs=3,verbose=True)
    
    
    #model.fit(X_train,y_train,epochs=1,batch_size=5)

    score=model.evaluate(X_test,y_test)

    print(score)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


training_step(X_train,y_train,X_test,y_test)

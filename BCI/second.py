from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from numpy import asarray
from matplotlib import pyplot
from numpy.random import randn
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, LeakyReLU, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from numpy import array, reshape, mean, std
from scipy.stats import zscore
from sklearn.preprocessing import OneHotEncoder
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.layers.merge import concatenate
import pickle
def fun():
    open_file = open("600s.pkl", "rb")
    d = pickle.load(open_file)
    open_file.close()
    
    num_classes = 6
    xtrain = []
    ytrain = []
    xtrain_images = []

    
    del d['Green_Heart']
    del d['Blue_Heart']

    del d['Green_Star']
    del d['Blue_Star']

    del d['Green_Rhombus']
    del d['Blue_Rhombus']

    del d['Green_Circle']
    del d['Blue_Circle']

    del d['Green_Square']
    del d['Blue_Square']

    del d['Green_Triangle']
    del d['Blue_Triangle']

    #print(len(d["Blue_Triangle"][0][0]))
    xtrain = []
    ytrain = []
    xtrain_images = []
    global revert_mapping
    #mapping = {"Red_Triangle" : 0,"Blue_Triangle":1, "Green_Triangle":2, "Red_Star":3,"Green_Star":4,"Blue_Star":5,"Red_Circle":6,"Green_Circle":7,"Blue_Circle":8,
    #           "Red_Heart":9,"Green_Heart":10,"Blue_Heart":11,"Red_Square":12,"Green_Square":13,"Blue_Square":14,"Red_Rhombus":15,"Green_Rhombus":16,"Blue_Rhombus":17}
    mapping = {"Red_Triangle" : 0,"Red_Heart":1, "Red_Rhombus":2,"Red_Circle":3,"Red_Star":4,"Red_Square":5}
    revert_mapping = {0 :"Red_Triangle\n", 1: "Red_Heart\n",2:"Red_Rhombus\n",3:"Red_Circle\n",4:"Red_Star\n",5:"Red_Square\n"}
    for i in d:
        for j in range(len(d[i])):
            xtrain.append([d[i][j]])
            ytrain.append(mapping[i])
            #print('./Dataset/temp1/'+i+'.jpg')
            #ad = os.getcwd()
            #os.chdir()
            ind_image = cv2.imread('./Datasets/temp1/'+i+'.jpg')
            #os.chdir(ad)
            ind_image = cv2.resize(ind_image,(28,28))
            ind_image = cv2.cvtColor(ind_image, cv2.COLOR_BGR2GRAY)
            xtrain_images.append(ind_image)
    '''
    for i in d:
        for j in range(len(d[i])):
            xtrain.append([d[i][j]])
            ytrain.append(i)
            ind_image = cv2.imread('./Dataset_basic/temp1/'+i+'.jpg')
            ind_image = cv2.resize(ind_image,(28,28))
            #ind_image = cv2.cvtColor(ind_image, cv2.COLOR_BGR2GRAY)
            xtrain_images.append(ind_image)
    '''
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    xtrain = xtrain.reshape(1760,8,600,1)
    xtrain /= 100000000
    '''
    for i in d:
        
        for j in range(len(d[i])):
            #cur = os.getcwd()
            #os.chdir('./Individual_images/'+i)
            #temp = next(os.walk('./Dataset/temp1'))[2]
            #print(temp)
            ind_image = cv2.imread('./Dataset/temp1/'+i+'.jpg')
            #os.chdir(cur)
            #print('Individual_images/'+ i + "/*.jpg")
            ind_image = cv2.resize(ind_image,(28,28))
            #ind_image = cv2.cvtColor(ind_image, cv2.COLOR_BGR2GRAY)
            xtrain_images.append(ind_image)
            xtrain.append(d[i][j])
            ytrain.append(i)
    # Load the dataset
    #(X_train, _), (_, _) = mnist.load_data()

    # Convert to float and Rescale -1 to 1 (Can also do 0 to 1)
    #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    '''
    #Add channels dimension. As the input to our gen and discr. has a shape 28x28x1.
    xtrain_images = np.array(xtrain_images)
    xtrain_images = xtrain_images.astype('float32')
    xtrain_images /= 255
    xtrain /= 255 
    #xtrain = np.array(xtrain)
    #print(xtrain.shape)
    #ytrain = np.array(ytrain)
    #X_train, X_test, Y_train, Y_test, X_train_I, X_test_I = train_test_split(xtrain,ytrain,xtrain_images,test_size = 0.04444444, shuffle = True)
    c1 = 0
    #X_train = xtrain.astype('float32')
    #X_test = X_test.astype('float32')
    #X_train /= 255
    #X_test /= 255
    global X_train, Y_train
    X_train = xtrain
    Y_train = ytrain
    X_train_I = xtrain_images
    #print("Here",X_train.shape)
    #X_train = np.expand_dims(X_train, axis=3) 

    #half_batch = 16#int(batch_size / 2)
    #print('half_batch')
    #print(half_batch)
    #X_train_I = np.expand_dims(X_train_I, axis=3)
    eeg_model = load_model('eeg_descriptions_shapes.h5')
    eeg_for_gen = eeg_model.predict(X_train)
fun()
def one(epoch):
    global X_train, Y_train, mapping, revert_mapping
    new_xtrain = X_train[:36]
    new_ytrain = Y_train[:36]
    f = open('Chikoo/basic/mnist_'+str(epoch)+'.txt','w')
    add_to_text = []
    for i in new_ytrain:
        add_to_text.append(revert_mapping[i])
    f.writelines(add_to_text)
    r, c = 6, 6
    noise = np.random.normal(0, 1, (36, 6))
    eeg_model = load_model('eeg_descriptions_shapes.h5')
    eeg_for_gen = eeg_model.predict(new_xtrain)
    generator = load_model('generator_model_basic_shapes_red_50x50_1.h5')
    gen_imgs = generator.predict([noise,eeg_for_gen])
    tempo = []
    
    for i in range(len(gen_imgs)):
        tempo.append([])
        for j in range(len(gen_imgs[i])):
            tempo[i].append([])
            for k in range(len(gen_imgs[i][j])):
                tempo[i][j].append([gen_imgs[i][j][k], 0, 0])
    print(gen_imgs[0])
    plt.imshow(gen_imgs[0])
    plt.show()
    '''tempo = np.array(tempo)
    tempo = tempo.astype('float32')
    gen_imgs = tempo*200                
    #print(np.array(tempo).shape)
    #gen_imgs *=255
    #print((gen_imgs[0]))
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("Chikoo/basic_t2/mnist_%d.png" % epoch)'''
    plt.close()
    
def two(epoch):
    global X_train
    new_xtrain = X_train[1000:1036]
    global Y_train, revert_mapping
    new_ytrain = Y_train[1000:1036]
    f = open('Chikoo/basic_t2/mnist_'+str(epoch)+'.txt','w')
    add_to_text = []
    for i in new_ytrain:
        add_to_text.append(revert_mapping[i])
    f.writelines(add_to_text)
    r, c = 6, 6
    noise = np.random.normal(0, 1, (36, 6))
    eeg_model = load_model('eeg_descriptions_shapes.h5')
    eeg_for_gen = eeg_model.predict(new_xtrain)
    generator = load_model('generator_model_basic_shapes_red_50x50_1.h5')
    gen_imgs = generator.predict([noise,eeg_for_gen])
    tempo = []
    
    for i in range(len(gen_imgs)):
        tempo.append([])
        for j in range(len(gen_imgs[i])):
            tempo[i].append([])
            for k in range(len(gen_imgs[i][j])):
                tempo[i][j].append([gen_imgs[i][j][k], 0, 0])
    print(gen_imgs[0])
    plt.imshow(gen_imgs[0])
    plt.show()
    '''tempo = np.array(tempo)
    tempo = tempo.astype('float32')
    gen_imgs = tempo*200 
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("Chikoo/basic_t2/mnist_%d.png" % epoch)'''
    plt.close()
def three(epoch):
    global X_train
    new_xtrain = X_train[900:936]
    global Y_train, revert_mapping
    new_ytrain = Y_train[900:936]
    f = open('Chikoo/basic_t2/mnist_'+str(epoch)+'.txt','w')
    add_to_text = []
    for i in new_ytrain:
        add_to_text.append(revert_mapping[i])
    f.writelines(add_to_text)
    r, c = 6, 6
    noise = np.random.normal(0, 1, (36, 6))
    eeg_model = load_model('eeg_descriptions_shapes.h5')
    eeg_for_gen = eeg_model.predict(new_xtrain)
    generator = load_model('generator_model_basic_shapes_red_50x50_1.h5')
    gen_imgs = generator.predict([noise,eeg_for_gen])
    tempo = []
    
    for i in range(len(gen_imgs)):
        tempo.append([])
        for j in range(len(gen_imgs[i])):
            tempo[i].append([])
            for k in range(len(gen_imgs[i][j])):
                tempo[i][j].append([gen_imgs[i][j][k], 0, 0])
    print(gen_imgs[0])
    plt.imshow(gen_imgs[0])
    plt.show()
    '''tempo = np.array(tempo)
    tempo = tempo.astype('float32')
    gen_imgs = tempo *200
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("Chikoo/basic_t2/mnist_%d.png" % epoch)'''
    plt.close()
def four(epoch):
    global X_train
    new_xtrain = X_train[800:836]
    global Y_train, revert_mapping
    new_ytrain = Y_train[800:836]
    f = open('Chikoo/basic_t2/mnist_'+str(epoch)+'.txt','w')
    add_to_text = []
    for i in new_ytrain:
        add_to_text.append(revert_mapping[i])
    f.writelines(add_to_text)
    r, c = 6, 6
    noise = np.random.normal(0, 1, (36, 6))
    eeg_model = load_model('eeg_descriptions_shapes.h5')
    eeg_for_gen = eeg_model.predict(new_xtrain)
    generator = load_model('generator_model_basic_shapes_red_50x50_1.h5')
    updated = [eeg_for_gen[i] for i in range(len(eeg_for_gen))  if i < 16]
    gen_imgs = generator.predict([noise,eeg_for_gen])
    tempo = []
    
    for i in range(len(gen_imgs)):
        tempo.append([])
        for j in range(len(gen_imgs[i])):
            tempo[i].append([])
            for k in range(len(gen_imgs[i][j])):
                tempo[i][j].append([gen_imgs[i][j][k], 0, 0])
    tempo = np.array(tempo)
    tempo = tempo.astype('float32')
    print(gen_imgs[0])
    plt.imshow(gen_imgs[0])
    plt.show()
    '''gen_imgs = tempo *200
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("Chikoo/basic_t2/mnist_%d.png" % epoch)'''
    plt.close()
def five(epoch):
    global X_train
    new_xtrain = X_train[144:180]
    global Y_train, revert_mapping
    new_ytrain = Y_train[144:180]
    f = open('Chikoo/basic_t2/mnist_'+str(epoch)+'.txt','w')
    add_to_text = []
    for i in new_ytrain:
        add_to_text.append(revert_mapping[i])
    f.writelines(add_to_text)
    r, c = 6, 6
    noise = np.random.normal(0, 1, (36, 6))
    eeg_model = load_model('eeg_descriptions_shapes.h5')
    eeg_for_gen = eeg_model.predict(new_xtrain)
    updated = [eeg_for_gen[i] for i in range(len(eeg_for_gen))  if i < 16]
    generator = load_model('generator_model_basic_shapes_red_50x50_1.h5')
    gen_imgs = generator.predict([noise,eeg_for_gen])
    tempo = []
    
    for i in range(len(gen_imgs)):
        tempo.append([])
        for j in range(len(gen_imgs[i])):
            tempo[i].append([])
            for k in range(len(gen_imgs[i][j])):
                tempo[i][j].append([gen_imgs[i][j][k], 0, 0])
    tempo = np.array(tempo)
    tempo = tempo.astype('float32')
    print(gen_imgs[0])
    plt.imshow(gen_imgs[0])
    plt.show()
    '''gen_imgs = tempo *200
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("Chikoo/basic_t2/mnist_%d.png" % epoch)'''
    plt.close()
def six(epoch):
    global X_train
    new_xtrain = X_train[480:516]
    global Y_train, revert_mapping
    new_ytrain = Y_train[480:516]
    f = open('Chikoo/basic_t2/mnist_'+str(epoch)+'.txt','w')
    add_to_text = []
    for i in new_ytrain:
        add_to_text.append(revert_mapping[i])
    f.writelines(add_to_text)
    r, c = 6, 6
    noise = np.random.normal(0, 1, (36, 6))
    eeg_model = load_model('eeg_descriptions_shapes.h5')
    eeg_for_gen = eeg_model.predict(new_xtrain)
    updated = [eeg_for_gen[i] for i in range(len(eeg_for_gen))  if i < 16]
    generator = load_model('generator_model_basic_shapes_red_50x50_1.h5')
    gen_imgs = generator.predict([noise,eeg_for_gen])
    tempo = []
    
    for i in range(len(gen_imgs)):
        tempo.append([])
        for j in range(len(gen_imgs[i])):
            tempo[i].append([])
            for k in range(len(gen_imgs[i][j])):
                tempo[i][j].append([gen_imgs[i][j][k], 0, 0])
    tempo = np.array(tempo)
    tempo = tempo.astype('float32')
    print(gen_imgs[0])
    plt.imshow(gen_imgs[0])
    plt.show()
    '''gen_imgs = tempo *200
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("Chikoo/basic_t2/mnist_%d.png" % epoch)'''
    plt.close()
def seven(epoch):
    global X_train
    new_xtrain = X_train[216:252]
    global Y_train, revert_mapping
    new_ytrain = Y_train[216:252]
    f = open('Chikoo/basic_t2/mnist_'+str(epoch)+'.txt','w')
    add_to_text = []
    for i in new_ytrain:
        add_to_text.append(revert_mapping[i])
    f.writelines(add_to_text)
    r, c = 6, 6
    noise = np.random.normal(0, 1, (36, 6))
    eeg_model = load_model('eeg_descriptions_shapes.h5')
    eeg_for_gen = eeg_model.predict(new_xtrain)
    updated = [eeg_for_gen[i] for i in range(len(eeg_for_gen))  if i < 16]
    generator = load_model('generator_model_basic_shapes_red_50x50_1.h5')
    gen_imgs = generator.predict([noise,eeg_for_gen])
    tempo = []
    
    for i in range(len(gen_imgs)):
        tempo.append([])
        for j in range(len(gen_imgs[i])):
            tempo[i].append([])
            for k in range(len(gen_imgs[i][j])):
                tempo[i][j].append([gen_imgs[i][j][k], 0, 0])
    tempo = np.array(tempo)
    tempo = tempo.astype('float32')
    print(gen_imgs[0])
    plt.imshow(gen_imgs[0])
    plt.show()
    '''gen_imgs = tempo *200
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("Chikoo/basic_t2/mnist_%d.png" % epoch)'''
    plt.close()
def eight(epoch):
    global X_train
    new_xtrain = X_train[1152:1188]
    global Y_train, revert_mapping
    new_ytrain = Y_train[1152:1188]
    f = open('Chikoo/basic_t2/mnist_'+str(epoch)+'.txt','w')
    add_to_text = []
    for i in new_ytrain:
        add_to_text.append(revert_mapping[i])
    f.writelines(add_to_text)
    r, c = 6, 6
    noise = np.random.normal(0, 1, (36, 6))
    eeg_model = load_model('eeg_descriptions_shapes.h5')
    eeg_for_gen = eeg_model.predict(new_xtrain)
    updated = [eeg_for_gen[i] for i in range(len(eeg_for_gen))  if i < 16]
    generator = load_model('generator_model_basic_shapes_red_50x50_1.h5')
    gen_imgs = generator.predict([noise,eeg_for_gen])
    tempo = []
    
    for i in range(len(gen_imgs)):
        tempo.append([])
        for j in range(len(gen_imgs[i])):
            tempo[i].append([])
            for k in range(len(gen_imgs[i][j])):
                tempo[i][j].append([gen_imgs[i][j][k], 0, 0])
    tempo = np.array(tempo)
    tempo = tempo.astype('float32')
    print(gen_imgs[0])
    plt.imshow(gen_imgs[0])
    plt.show()
    '''gen_imgs = tempo *200
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("Chikoo/basic_t2/mnist_%d.png" % epoch)'''
    plt.close()

import matplotlib.image as npimg
def nine(epoch):
    global X_train
    new_xtrain = X_train[1388:1424]
    global Y_train, revert_mapping
    new_ytrain = Y_train[1388:1424]
    f = open('Chikoo/basic_t2/mnist_'+str(epoch)+'.txt','w')
    add_to_text = []
    for i in new_ytrain:
        add_to_text.append(revert_mapping[i])
    f.writelines(add_to_text)
    r, c = 6, 6
    noise = np.random.normal(0, 1, (36, 6))
    eeg_model = load_model('eeg_descriptions_shapes.h5')
    eeg_for_gen = eeg_model.predict(new_xtrain)
    updated = [eeg_for_gen[i] for i in range(len(eeg_for_gen))  if i < 16]
    generator = load_model('generator_model_basic_shapes_red_50x50_1.h5')
    gen_imgs = generator.predict([noise,eeg_for_gen])
    tempo = []
    
    for i in range(len(gen_imgs)):
        tempo.append([])
        for j in range(len(gen_imgs[i])):
            tempo[i].append([])
            for k in range(len(gen_imgs[i][j])):
                tempo[i][j].append([gen_imgs[i][j][k], 0, 0])
    tempo = np.array(tempo)
    tempo = tempo.astype('float32')
    #gen_imgs = tempo *200
    print(gen_imgs[0])
    plt.imshow(gen_imgs[0])
    plt.show()
    # Rescale images 0 - 1
    '''gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap = 'RdBu_r') #, cmap='gray'
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("Chikoo/basic_t2/mnist_%d.png" % epoch)'''
    plt.close()

def ten(epoch):
    global X_train
    new_xtrain = X_train[688:724]
    global Y_train, revert_mapping
    new_ytrain = Y_train[788:824]
    f = open('Chikoo/basic_t2/mnist_'+str(epoch)+'.txt','w')
    add_to_text = []
    for i in new_ytrain:
        add_to_text.append(revert_mapping[i])
    f.writelines(add_to_text)
    r, c = 6, 6
    noise = np.random.normal(0, 1, (36, 6))
    eeg_model = load_model('eeg_descriptions_shapes.h5')
    eeg_for_gen = eeg_model.predict(new_xtrain)
    updated = [eeg_for_gen[i] for i in range(len(eeg_for_gen))  if i < 16]
    generator = load_model('generator_model_basic_shapes_red_50x50_1.h5')
    gen_imgs = generator.predict([noise,eeg_for_gen])
    tempo = []
    
    for i in range(len(gen_imgs)):
        tempo.append([])
        for j in range(len(gen_imgs[i])):
            tempo[i].append([])
            for k in range(len(gen_imgs[i][j])):
                tempo[i][j].append([gen_imgs[i][j][k], 0, 0])
    tempo = np.array(tempo)
    tempo = tempo.astype('float32')
    #gen_imgs = tempo *200
    print(gen_imgs[20])
    plt.imshow(gen_imgs[20])
    plt.show()
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap = 'RdBu_r') #, cmap='gray'
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("Chikoo/basic_t2/mnist_%d.png" % epoch)
    plt.close()
#This function saves our images for us to view
one(0)
two(1)
three(2)
four(3)
five(4)
six(5)
seven(6)
eight(7)
nine(8)
ten(10)

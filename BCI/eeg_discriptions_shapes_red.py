import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
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
from keras.layers import Dense, Dropout, Reshape
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
print(tf.__version__)

global z
z = 0

'''
f = open('out1.txt')
d= {}
l = []

for i in range(10):
    d[str(i)] = [[]]
for i in range(5040):
    a = f.readline().split('\t')
    a[-1] = a[-1][13:]
    #print(a[-1])
    a[-1] = list(map(float,a[-1].split(',')))
    while(len(a[-1])!=272):
        a[-1].append(0)
    for blah in range(len(a[-1])):
        a[-1][blah] = [a[-1][blah]]
    if(len(d[a[4]][-1])==14):
        d[a[4]].append([])

    d[a[4]][-1].append(a[-1])


num_classes = 10
xtrain = []
ytrain = []

for i in d:
    for j in range(len(d[i])):        
        xtrain.append(d[i][j])
        ytrain.append(i)
'''    
#print(len(xtrain), " ; ", len(ytrain))

import pickle
import os
import numpy as np

open_file = open("600s.pkl", "rb")
d = pickle.load(open_file)
open_file.close()


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
#mapping = {"Red_Triangle" : 0,"Blue_Triangle":1, "Green_Triangle":2, "Red_Star":3,"Green_Star":4,"Blue_Star":5,"Red_Circle":6,"Green_Circle":7,"Blue_Circle":8,
#           "Red_Heart":9,"Green_Heart":10,"Blue_Heart":11,"Red_Square":12,"Green_Square":13,"Blue_Square":14,"Red_Rhombus":15,"Green_Rhombus":16,"Blue_Rhombus":17}
mapping = {"Red_Triangle" : 0,"Red_Heart":1, "Red_Rhombus":2,"Red_Circle":3,"Red_Star":4,"Red_Square":5}
for i in d:
    for j in range(len(d[i])):
        xtrain.append([d[i][j]])
        ytrain.append(mapping[i])

convert = []
print(len(ytrain))
for i in range(len(xtrain)):
    big_temp = []
    for j in range(len(xtrain[i][0])):
        temp = []
        for k in range(len(xtrain[i][0][j])):
            temp.append([xtrain[i][0][j][k]])
        big_temp.append(temp)
    convert.append(big_temp)


xtrain = np.array(convert)
ytrain = np.array(ytrain)
#print(xtrain[0][0])

print(xtrain.shape)
print("########################################## STOP HERE ###################################################3")
xtrain = xtrain.reshape(1760,8,600,1)
print(xtrain.shape)
print(set(ytrain))

xtrain = np.array(xtrain)
print(xtrain.shape)
ytrain = np.array(ytrain)
X_train, X_test, Y_train, Y_test = train_test_split(xtrain,ytrain,test_size = 0.1, shuffle = True)
#X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.142857, random_state=0, shuffle = True)
print()
c1 = 0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 100000000
X_test /= 100000000
L = len(X_train)
print(X_train.shape)
input_shape = (8,600,1)
ytrain = keras.utils.to_categorical(Y_train,num_classes= 6)
ytest = keras.utils.to_categorical(Y_test, num_classes = 6)
#yval = keras.utils.to_categorical(Y_val, 10)
model = Sequential()
model.add(Conv2D(48, kernel_size = (1,3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(1,3)))
model.add(Conv2D(24, kernel_size = (1, 5), activation = 'relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(1,3)))
model.add(Conv2D(48, kernel_size = (1, 5), activation = 'relu', padding = 'same'))#model.add(MaxPooling2D(pool_size=(1,3)))
model.add(Conv2D(48, kernel_size = (1, 3), activation = 'relu', padding = 'same'))
model.add(Dense(1000))
#model.add(Dense(2000))
#model.add(Conv2D(256, kernel_size = (1, 5), activation = 'relu', padding = 'same'))
#model.add(MaxPooling2D(pool_size=(1,3)))
#model.add(Conv2D(128, kernel_size = (1, 3), activation = 'relu', padding = 'same'))
model.add(Flatten())
#model.add(Dense(2352, activation='sigmoid'))
#model.add(Reshape((28,28,1)))
#smodel.add(Flatten())
model.add(Dense(6, activation='sigmoid'))
print(model.summary())
#for layer in model.layers:
#    print(layer.output_shape)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=0.00045, beta_1=0.72, beta_2=0.85, amsgrad=False),
              metrics=['accuracy'])
earlystopper = EarlyStopping(patience=50, verbose=1)
checkpointer = ModelCheckpoint('eeg_descriptions_shapes.h5', verbose=1, save_best_only=True, monitor = 'accuracy')
#history = model.fit(X_train, ytrain, batch_size = 36, epochs = 1000, callbacks = [earlystopper,checkpointer])
#model.save('eeg_descriptions_shapes_100.h5')


#f = open('test_file.txt')
#d= {}
#l = []

'''for i in range(10):
    d[str(i)] = [[]]
for i in range(5040-4032):
    a = f.readline().split('\t')
    a[-1] = a[-1][13:]
    #print(a[-1])
    a[-1] = list(map(float,a[-1].split(',')))
    while(len(a[-1])!=272):
        a[-1].append(0)
    for blah in range(len(a[-1])):
        a[-1][blah] = [a[-1][blah]]
    if(len(d[a[4]][-1])==14):
        d[a[4]].append([])

    d[a[4]][-1].append(a[-1])'''

xtest = []
ytest = []

'''for i in d:
    for j in range(len(d[i])):
        xtest.append(d[i][j])
        ytest.append(i)
X_test = np.array(xtest)
Y_test = np.array(ytest)
print(X_test.shape)'''
#from keras.models import Model
model= keras.models.load_model('eeg_descriptions_shapes.h5')

#layer_name = 'reshape'
#inter = Model(inputs= model.input, outputs = model.get_layer(layer_name).output)
#inter_out = inter.predict(X_test)
#print("shape +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
#print(inter_out.shape)

c1 = 0
predictions = model.predict_classes(X_test)
print(predictions.shape)
print(type(predictions),type(ytest))
for_eval_pred = []
for_eval_test = []
for i in range(len(Y_test)):
    for_eval_pred.append(int(Y_test[i]))
    for_eval_test.append(int(predictions[i]))
for i in range(len(predictions)):
    print('Predicted class : ', predictions[i], 'Actual class : ', Y_test[i])
    if(int(predictions[i]) == int(Y_test[i])):
        c1+=1

print()
print("******************************* ACCURACY : ",c1/len(Y_test))
#print('Accuracy : ', c1/len(predictions))
# plot loss during training
#plt.subplot(211)
#plt.title('Loss')
#plt.plot(history.history['loss'], label='train')

#plt.legend()
# plot accuracy during training
#plt.subplot(212)
#plt.title('Accuracy')
#plt.plot(history.history['accuracy'], label='train')

#plt.legend()
#plt.show()

# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)
accuracy = accuracy_score(for_eval_test, for_eval_pred)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(for_eval_test, for_eval_pred, average = 'weighted')
print('Precision: %.3f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(for_eval_test, for_eval_pred, average = 'weighted')
print('Recall: %.3f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(for_eval_test, for_eval_pred, average = 'weighted')
print('F1 score: %.3f' % f1)

results = confusion_matrix(for_eval_test, for_eval_pred)
print(results)
#model.save('../h5 files/model0.h5')


'''
print((X_train[0]).shape)
print((X_train[0]).shape[0])
model.add(Dense(128, input_dim=(X_train[0]).shape, activation='relu', kernel_initializer='normal'))
model.add(Dense(256, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.05))
model.add(Dense(256, activation='relu', kernel_initializer='normal'))
model.add(Dropout(0.05))
model.add(Dense(256, activation='relu', kernel_initializer='normal'))
model.add(Dense(1, activation='linear', kernel_initializer='normal'))
print(model.summary())
es = EarlyStopping(monitor='val_mean_absolute_error', min_delta=0.001, patience=20)
adam = Adam(clipnorm=1.)
model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['mean_absolute_error'])
history = model.fit(X_train, ytrain, batch_size=128, epochs=50, verbose=1, validation_data=(X_val, yval), callbacks=[es])
predictions = model.predict(X_test)
'''

import pickle
import os
import numpy as np
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

open_file = open("600s.pkl", "rb")
d = pickle.load(open_file)
open_file.close()


del d['Blue_Heart']
del d['Green_Heart']

del d['Blue_Star']
del d['Green_Star']

del d['Blue_Rhombus']
del d['Green_Rhombus']

del d['Blue_Circle']
del d['Green_Circle']

del d['Blue_Square']
del d['Green_Square']

del d['Blue_Triangle']
del d['Green_Triangle']

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

blue_triangle_train, blue_triangle_test =  train_test_split([xtrain[i] for i in range(len(xtrain)) if ytrain[i] == 0],test_size = 0.15, shuffle = True)
blue_heart_train, blue_heart_test =  train_test_split([xtrain[i] for i in range(len(xtrain)) if ytrain[i] == 1],test_size = 0.15, shuffle = True)
blue_rhombus_train, blue_rhombus_test =  train_test_split([xtrain[i] for i in range(len(xtrain)) if ytrain[i] == 2],test_size = 0.15, shuffle = True)
blue_circle_train, blue_circle_test =  train_test_split([xtrain[i] for i in range(len(xtrain)) if ytrain[i] == 3],test_size = 0.15, shuffle = True)
blue_star_train, blue_star_test =  train_test_split([xtrain[i] for i in range(len(xtrain)) if ytrain[i] == 4],test_size = 0.15, shuffle = True)
blue_square_train, blue_square_test =  train_test_split([xtrain[i] for i in range(len(xtrain)) if ytrain[i] == 5],test_size = 0.15, shuffle = True)

blue_triangle_train_file = open('green_triangle_train.pickle','wb')
pickle.dump(blue_triangle_train, blue_triangle_train_file)

blue_triangle_test_file = open('green_triangle_test.pickle','wb')
pickle.dump(blue_triangle_test, blue_triangle_test_file)

blue_heart_train_file = open('green_heart_train.pickle','wb')
pickle.dump(blue_heart_train, blue_heart_train_file)

blue_heart_test_file = open('green_heart_test.pickle','wb')
pickle.dump(blue_heart_test, blue_heart_test_file)

blue_rhombus_train_file = open('green_rhombus_train.pickle','wb')
pickle.dump(blue_rhombus_train, blue_rhombus_train_file)

blue_rhombus_test_file = open('green_rhombus_test.pickle','wb')
pickle.dump(blue_rhombus_test, blue_rhombus_test_file)

blue_circle_train_file = open('green_circle_train.pickle','wb')
pickle.dump(blue_circle_train, blue_circle_train_file)

blue_circle_test_file = open('green_circle_test.pickle','wb')
pickle.dump(blue_circle_test, blue_circle_test_file)

blue_star_train_file = open('green_star_train.pickle','wb')
pickle.dump(blue_star_train, blue_star_train_file)

blue_star_test_file = open('green_star_test.pickle','wb')
pickle.dump(blue_star_test, blue_star_test_file)

blue_square_train_file = open('green_square_train.pickle','wb')
pickle.dump(blue_square_train, blue_square_train_file)

blue_square_test_file = open('green_square_test.pickle','wb')
pickle.dump(blue_square_test, blue_square_test_file)

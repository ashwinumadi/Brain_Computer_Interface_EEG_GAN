import pickle
import os
import numpy as np

open_file = open("600s.pkl", "rb")
d = pickle.load(open_file)
open_file.close()

print(len(d["Blue_Triangle"][0][0]))
xtrain = []
ytrain = []

for i in d:
    for j in range(len(d[i])):
        xtrain.append([d[i][j]])
        ytrain.append(i)


xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

print(xtrain.shape)

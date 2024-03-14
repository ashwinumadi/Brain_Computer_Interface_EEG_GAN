
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
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
            ind_image = cv2.imread('./Datasets/temp1/'+i+'.jpg')
            ind_image = cv2.resize(ind_image,(28,28))
            ind_image = cv2.cvtColor(ind_image, cv2.COLOR_BGR2GRAY)
            xtrain_images.append(ind_image)
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)
    xtrain = xtrain.reshape(1760,8,600,1)
    xtrain /= 100000000
    #Add channels dimension. As the input to our gen and discr. has a shape 28x28x1.
    xtrain_images = np.array(xtrain_images)
    xtrain_images = xtrain_images.astype('float32')
    xtrain_images /= 255
    xtrain /= 255
    c1 = 0
    global X_train, Y_train
    X_train = xtrain
    Y_train = ytrain
    X_train_I = xtrain_images
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
    plt.close()

#This function saves our images for us to view
one(0)

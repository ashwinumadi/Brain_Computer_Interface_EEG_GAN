from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import keras
from keras.layers import Dense, Flatten, BatchNormalization, LeakyReLU, Reshape
from keras import backend as K
import cv2
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.layers.merge import concatenate
import pickle
 

img_rows = 50
img_cols = 50
channels = 1
img_shape = (img_rows, img_cols, channels)

##########################################################################
#Given input of noise (latent) vector, the Generator produces an image.
def build_generator():
    noise_shape = (6,) #1D array of size 100 (latent vector / noise)
    eeg_shape = (6,)
    inputsA = Input(noise_shape)
    inputsB = Input(eeg_shape)
    s = concatenate([inputsA, inputsB])
    d1 = Dense(256)(s)
    l1 = LeakyReLU(alpha = 0.8)(d1)
    b1 = BatchNormalization(momentum = 0.8)(l1)
    d2 = Dense(512) (b1)
    l2 = LeakyReLU(alpha = 0.2)(d2)
    b2 = BatchNormalization(momentum = 0.8)(l2)
    d3 = Dense(1024) (b2)
    l3 = LeakyReLU(alpha = 0.2)(d3)
    b3 = BatchNormalization(momentum = 0.8)(l3)
    d4 = Dense(1024) (b3)
    l4 = LeakyReLU(alpha = 0.2)(d4)
    b4 = BatchNormalization(momentum = 0.8)(l4)
    d5 = Dense(np.prod(img_shape), activation = 'tanh')(b4)
    output = Reshape(img_shape)(d5)
    model = Model(inputs = [inputsA, inputsB], outputs = [output])
    model.summary()
    return model


def build_discriminator():
    InputsA = Input(img_shape)
    f1 = Flatten()(InputsA)
    d1 = Dense(512)(f1)
    l1 = LeakyReLU(alpha = 0.2)(d1)
    d2 = Dense(256)(l1)
    l2 = LeakyReLU(alpha = 0.2)(d2)
    output = Dense(7, activation = 'sigmoid')(l2)
    model = Model(inputs = [InputsA], outputs = [output])
    model.summary()
    return model

#images and setting the ground truths. 
def train(epochs, batch_size=128, save_interval=50):
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
    mapping = {"Red_Triangle" : 0,"Red_Heart":1, "Red_Rhombus":2,"Red_Circle":3,"Red_Star":4,"Red_Square":5}
    r = {0:"Red_Triangle",1:"Red_Heart",2: "Red_Rhombus",3:"Red_Circle",4:"Red_Star",5:"Red_Square"}

    open_file = open("green_triangle_train.pickle", "rb")
    d = pickle.load(open_file)
    print(np.array(d).shape)
    xtrain= d
    ytrain  = [0 for i in range(len(d))]
    open_file.close()



    open_file = open("green_heart_train.pickle", "rb")
    d = pickle.load(open_file)
    xtrain.extend(d)
    print(np.array(xtrain).shape)
    ytrain.extend([1 for i in range(len(d))])
    open_file.close()



    open_file = open("green_rhombus_train.pickle", "rb")
    d = pickle.load(open_file)
    xtrain.extend(d)
    ytrain.extend([2 for i in range(len(d))])
    open_file.close()



    open_file = open("green_circle_train.pickle", "rb")
    d = pickle.load(open_file)
    xtrain.extend(d)

    ytrain.extend([3 for i in range(len(d))])
    open_file.close()



    open_file = open("green_star_train.pickle", "rb")
    d = pickle.load(open_file)
    xtrain.extend(d)
    ytrain.extend([4 for i in range(len(d))])
    open_file.close()



    open_file = open("green_square_train.pickle", "rb")
    d = pickle.load(open_file)
    xtrain.extend(d)
    ytrain.extend([5 for i in range(len(d))])
    open_file.close()

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
    print('convert : ', len(convert), len(convert[0]), len(convert[0][0]), len(convert[0][0][0]))
    xtrain_images = []
    for i in ytrain:
            ind_image = cv2.imread('./Datasets/temp1/'+r[i]+'.jpg')
            #os.chdir(ad)
            ind_image = cv2.resize(ind_image,(50,50))
            ind_image = cv2.cvtColor(ind_image, cv2.COLOR_BGR2GRAY)
            xtrain_images.append(ind_image)

    xtrain = np.array(convert)
    fake_validity = [6 for i in range(1496)]
    for i in range(len(fake_validity)):
        ytrain.append(fake_validity[i])
    ytrain = np.array(ytrain)
    print(ytrain)
    print('here :',xtrain.shape)
    ytrain =ytrain.reshape((1496*2,1))
    ytrain = keras.utils.to_categorical(ytrain,num_classes= 7)
    xtrain /= 100000000
    #Add channels dimension. As the input to our gen and discr. has a shape 28x28x1.
    xtrain_images = np.array(xtrain_images)
    xtrain_images = xtrain_images.astype('float32')
    xtrain_images /= 255
    xtrain /= 255 
    c1 = 0
    global X_train
    X_train = xtrain
    X_train_I = xtrain_images
    eeg_model = load_model('eeg_descriptions_shapes.h5')
    eeg_for_gen = eeg_model.predict(X_train)


#We then loop through a number of epochs to train our Discriminator by first selecting
#a random batch of images from our true dataset, generating a set of images from our
#Generator, feeding both set of images into our Discriminator, and finally setting the
#loss parameters for both the real and fake images, as well as the combined loss. 
    
    for epoch in range(epochs):

        noise = np.random.normal(0, 1, (1496, 6))
        # Generate a half batch of fake images
        gen_imgs = generator.predict([noise,eeg_for_gen])
        # Train the discriminator on real and fake images, separately
        d_loss_real = discriminator.train_on_batch([X_train_I], ytrain[:1496])
        
        #fake_validity = fake_validity.reshape((1760,1))
        d_loss_fake = discriminator.train_on_batch([gen_imgs], ytrain[1496:])
        #print(d_loss_fake)
    #take average loss from real and fake images. 
    #
        #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) 

#And within the same loop we train our Generator, by setting the input noise and
#ultimately training the Generator to have the Discriminator label its samples as valid
#by specifying the gradient loss.
        # ---------------------
        #  Train Generator
        # ---------------------
#Create noise vectors as input for generator. 
#Create as many noise vectors as defined by the batch size. 
#Based on normal distribution. Output will be of size (batch size, 100)
        #batch_size = 16
        noise = np.random.normal(0, 1, (1496, 6)) 
        #print(noise.shape)
        # The generator wants the discriminator to label the generated samples
        # as valid (ones)
        #This is where the genrator is trying to trick discriminator into believing
        #the generated image is true (hence value of 1 for y)
        valid_y = np.array([1] * 1496) #Creates an array of all ones of size=batch size
        #print(valid_y.shape)
        # Generator is part of combined where it got directly linked with the discriminator
        # Train the generator with noise as x and 1 as y. 
        # Again, 1 as the output as it is adversarial and if generator did a great
        #job of folling the discriminator then the output would be 1 (true)
        #checkpointer = ModelCheckpoint('generator_model_basic_shapes_1.h5', verbose=1, save_best_only=False)
        g_loss = combined.train_on_batch([noise,eeg_for_gen], ytrain[:1496])


#Additionally, in order for us to keep track of our training process, we print the
#progress and save the sample image output depending on the epoch interval specified.  
# Plot the progress
        
        #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        print(epoch)
        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            save_imgs(epoch)

#when the specific sample_interval is hit, we call the
#sample_image function. Which looks as follows.

def save_imgs(epoch):
    global X_train
    new_xtrain = X_train[:16]
    r, c = 4, 4
    noise = np.random.normal(0, 1, (16, 6))
    eeg_model = load_model('eeg_descriptions_shapes.h5')
    #print(new_xtrain.shape)
    eeg_for_gen = eeg_model.predict(new_xtrain)
    updated = [eeg_for_gen[i] for i in range(len(eeg_for_gen))  if i < 16]
    #print(noise.shape, eeg_for_gen.shape)
    gen_imgs = generator.predict([noise,eeg_for_gen])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("GAN_basic_shapes/gray/mnist_%d.png" % epoch)
    plt.close()
#This function saves our images for us to view

##############################################################################

#Let us also define our optimizer for easy use later on.
#That way if you change your mind, you can change it easily here
optimizer = Adam(0.0002, 0.5)  #Learning rate and momentum.

# Build and compile the discriminator first. 
#Generator will be trained as part of the combined model, later. 
#pick the loss function and the type of metric to keep track.                 
#Binary cross entropy as we are doing prediction and it is a better
#loss function compared to MSE or other. 
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

#build and compile our Discriminator, pick the loss function

#SInce we are only generating (faking) images, let us not track any metrics.
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

##This builds the Generator and defines the input noise. 
#In a GAN the Generator network takes noise z as an input to produce its images.  
z = Input(shape=(6,))
y_gen = Input(shape=(6,)) #Our random input to the generator
img = generator([z,y_gen])

#This ensures that when we combine our networks we only train the Generator.
#While generator training we do not want discriminator weights to be adjusted. 
#This Doesn't affect the above descriminator training.     
discriminator.trainable = False  

#This specifies that our Discriminator will take the images generated by our Generator
#and true dataset and set its output to a parameter called valid, which will indicate
#whether the input is real or not.
y_dis= Input(shape = (28,28,1))
valid = discriminator(img)  #Validity check on the generated image


#Here we combined the models and also set our loss function and optimizer. 
#Again, we are only training the generator here. 
#The ultimate goal here is for the Generator to fool the Discriminator.  
# The combined model  (stacked generator and discriminator) takes
# noise as input => generates images => determines validity

combined = Model([z,y_gen], valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)


train(epochs=12000, batch_size=360, save_interval=100)

#Save model for future use to generate fake images
#Not tested yet... make sure right model is being saved..
#Compare with GAN4

generator.save('generator_model_basic_shapes_red_50x50_1.h5')  #Test the model on GAN4_predict...
#Change epochs back to 30K
                
#Epochs dictate the number of backward and forward propagations, the batch_size
#indicates the number of training samples per backward/forward propagation, and the
#sample_interval specifies after how many epochs we call our sample_image function.



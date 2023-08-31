import requests
import numpy as np
import pandas as pd
import copy
import time
import os
import cv2
from matplotlib import *
use('Agg')
from matplotlib import pyplot as plt
from keras.models import load_model
from flask import Flask,render_template, redirect, request, render_template_string

def plot(shape, colour):
        Pz = []
        Oz = []
        P3 = []
        P4 = []
        PO7 = []
        PO8 = []
        O1 = []
        O2 = []
        cwd = os.getcwd()
        print(cwd)
        file_name = cwd +'/repo/' + shape + "_" + colour + ".easy"
        image_path = cwd + '/repo/' + shape + "_" + colour + ".jpg"
        print(shape, colour)
        print()
        f = open(file_name, "r")
        count = 0
        while(True):
            a = f.readline().split()
            count+=1
            if(len(a) == 0):
                break
            a = list(map(float,a))
            if(count>1799 and count<2400):
                    Pz.append(a[0])
                    Oz.append(a[1])
                    P3.append(a[2])
                    P4.append(a[3])
                    PO7.append(a[4])
                    PO8.append(a[5])
                    O1.append(a[6])
                    O2.append(a[7])

        figure, blah = plt.subplots(8,2)
        gs = blah[7, 1].get_gridspec()

        image = plt.imread(image_path)

        for ax in blah[:8, -1]:
            ax.remove()
        axbig = figure.add_subplot(gs[2:6, -1])
        blah[0][0].plot(Pz)
        blah[1][0].plot(Oz)
        blah[2][0].plot(P3)
        blah[3][0].plot(P4)
        blah[4][0].plot(PO7)
        blah[5][0].plot(PO8)
        blah[6][0].plot(O1)
        blah[7][0].plot(O2)
        axbig.imshow(image)
        plt.savefig("./static/image/blah.png")
        


def reconstruct(shape, colour):
    
        Pz = []
        Oz = []
        P3 = []
        P4 = []
        PO7 = []
        PO8 = []
        O1 = []
        O2 = []
        cwd = os.getcwd()
        print(cwd)
        file_name = cwd +'/repo/' + shape + "_" + colour + ".easy"
        image_path = cwd + '/repo/' + shape + "_" + colour + ".jpg"
        f = open(file_name, "r")
        count = 0
        while(True):
            a = f.readline().split()
            count+=1
            if(len(a) == 0):
                break
            a = list(map(float,a))
            if(count>1199 and count<1800):
                    Pz.append(a[0])
                    Oz.append(a[1])
                    P3.append(a[2])
                    P4.append(a[3])
                    PO7.append(a[4])
                    PO8.append(a[5])
                    O1.append(a[6])
                    O2.append(a[7])
                    
        nonsense = [Pz, Oz, P3, P4, PO7, PO8, O1, O2]
        for i in nonsense:
                for j in range(len(i)):
                        i[j] = [i[j]]
        new_xtrain = np.array([nonsense])
        noise = np.random.normal(0, 1, (1, 6))
        eeg_model = load_model('eeg_descriptions_shapes_blue.h5')
        eeg_for_gen = eeg_model.predict(new_xtrain)
        generator = load_model('generator_model_basic_shapes_red_6_op_t3.h5')
        gen_imgs = generator.predict([noise,eeg_for_gen])
        #print(gen_imgs)
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(2,2)
        cnt = 0
        for i in range(1):
                for j in range(1):
                        axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')

                        axs[i,j].axis('off')
                        cnt += 0
        global harsh
        plt.savefig("./static/image/vayy.png")
        plt.close()
        

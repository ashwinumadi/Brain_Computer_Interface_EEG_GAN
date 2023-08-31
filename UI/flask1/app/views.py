import requests
import numpy as np
import pandas as pd
import copy
import time
import os
from matplotlib import *
from matplotlib import pyplot as plt
from flask import Flask,render_template, redirect, request, render_template_string
import functions

shape = ""
colour = ""

app = Flask(__name__)

@app.route('/')
def index():
        return render_template('real_basic.html')

@app.route('/start', methods=['POST','GET'])
def basic():
        return render_template('basic.html')


@app.route('/start/submit', methods=['POST','GET'])
def redirecting():
        sp = request.form.get("shape")
        cl = request.form.get("colour")
        global shape
        shape = sp
        global colour
        colour = cl
        print("glo : ",shape, colour)
        print("lo : ",sp,cl)
        functions.plot(shape, colour)
        return render_template("plot.html")


@app.route('/start/submit/reconstruct', methods=['POST','GET'])
def img_reconstruct():
        global shape
        global colour
        print("From reconstruct : ", shape, colour)
        functions.reconstruct(shape, colour)
        
        return render_template("reconstructed.html")


@app.route('/start1', methods=['POST','GET'])
def render_plate():
    return 0


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = '3000')

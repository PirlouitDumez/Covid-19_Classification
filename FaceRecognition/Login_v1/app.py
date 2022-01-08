from flask.app import *
from flask.templating import render_template
from flask.wrappers import Response
from requests.api import request
import tensorflow as tf
from keras.layers import Dense
from flask import *
import cv2
import pyautogui
import time
import numpy as np
import pickle
#import sqlite3
#import click
#import socket
import pickle
#import struct ## new
import mysql.connector
import webbrowser
from login import *


### GLOBALS
app = Flask(__name__)
DATASET_FILE = 'Recognition/dataset/Register_faces'
DATASET = []
model_detection, model_rpn, faceNet, sklearn_model, DATASET = loadVar(model_detection_file='Detection/models/best_detector_inception.h5',
                                                       model_rpn_file='Detection/RPN/models/last_RPN_workstation_output.h5',
                                                       model_reco_file="Recognition/models/last_facenet_simple_encoder_bis_transfer1.h5",
                                                       #sklearn_file='Recognition/models/model_sklearn_random_forest.sav',
                                                       sklearn_file='euclidian',
                                                       dataset_file=DATASET_FILE)
TRESHOLD = 10
k = 20
FACES = []


###mydb = mysql.connector.connect (host="localhost", user="root", passwd="camion38", database="campusdufutur")

app . config ['SECRET_KEY'] =  'vnkdjnfjknfl1232 #'

# Open Camera object

camera = cv2.VideoCapture(0)




def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame,face = showFaces(frame, model_rpn, model_detection=None)
            FACES.append(face)
            #FACES = FACES[-10:]
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gohome')
def gohome():
    return render_template('home.html')

@app.route('/gologin', methods=['POST','GET'])
def gologin():
    if request.method == 'GET':
        return render_template('login.html')
    else :
        return render_template('home.html')


@app.route('/gocreation')
def gocreation():
    global FACES
    FACES = []
    return render_template('create.html')



@app.route('/login', methods=['POST'])
def login():
    global FACES, k, DATASET
    email = request.form['email']

    reco,dists = testFaces(email, FACES[-1], DATASET, treshold=TRESHOLD,
                     model_detection=model_detection, model_rpn=model_rpn, faceNet=faceNet, sklearn_model=sklearn_model,
                     k=k, show=False)
    names = recognizeFace(FACES[-1], DATASET, faceNet, sklearn_model, treshold=.5)
    FACES = FACES[-5:]
    
##    if email==names[0]:
##        return render_template('Page_Acceuil.html')
##    else:
##        return render_template('login.html')
        
    
##    if reco:
##        return "GOOD : "+str(dists)+str(names)#render_template('create.html')
##    return "NOT GOOD : "+str(dists)+str(names)#render_template('index.html')
    treshold = 1
##    if int(dists[0]) < treshold:
##        return "GOOD : "+str(dists)+str(names)#render_template('create.html')
##    return "NOT GOOD : "+str(dists)+str(names)#render_template('index.html')
    if reco:
        if int(dists[0]) < treshold:
            return render_template('home.html')
        return render_template('login.html')
    return render_template('login.html')




##@app.route('/creation', methods=['POST'])
##def creation():
##    login = request.form['login']
##    mdp = request.form['Mdp']
##    nom = request.form['nom']
##    prenom = request.form['prenom']
##    Id = request.form['Id']
##
##    return render_template('create.html')


@app.route('/register', methods=['POST'])
def register():
    global FACES, DATASET_FILE, DATASET, faceNet
    identifiant = request.form['identifiant']
    if isInvited(identifiant):
        store_faces_in_file(FACES, identifiant, DATASET_FILE)
        DATASET = update_dataset(DATASET, identifiant, faceNet, FACES)
        delete_invitation(identifiant)
        #return str([identifiant, [len(i) for i in FACES]])
        return render_template('login.html')
    else:
        return render_template('create.html')


if  __name__  ==  '__main__' :
    webbrowser.open('http://127.0.0.1:5000/')

    app.run ( debug = True )

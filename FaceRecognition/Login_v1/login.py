import cv2
import numpy as np
import time
import keras.backend as K
import os
import random
from scipy.spatial import distance
import pickle

from Detection.cropImages import *
from Detection.detectFaces import *




def sklearn_distance(emb1, emb2, sklearn_model):
    x = emb1-emb2
    return sklearn_model.predict([x])[0]


def euclidean_distance(vectors):
    (featsA, featsB) = vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,keepdims=True)
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def simulateEntryFile(encoder, folder='dataset/trainset_webcam'):
    """Créer le dataset d'entrée pour recognizeFaces"""
    dataset = {}
    files = os.listdir(folder)
    #print(encoder.summary())

    for file in files:
        name = file.split('_')[0]
        im = cv2.imread(folder + '/' + file)
        if name not in dataset.keys():
            dataset[name] = [encoder.predict(np.array([cv2.resize(im, (128,128))/255]))[0]]
        else:
            dataset[name].append(encoder.predict(np.array([cv2.resize(im, (128,128))/255]))[0])
    pickle.dump(dataset, open('pickle_dataset.p','wb'))
    return dataset

def load_dataset(file_name):
    return  pickle.load(open('pickle_dataset.p','rb'))


def loadVar(model_detection_file='Detection/models/best_detector_inception.h5',
            model_rpn_file='Detection/RPN/models/last_RPN_workstation_output.h5',
            model_reco_file="Recognition/models/last_facenet_simple_encoder_bis_transfer1.h5",
            dataset_file='Recognition/dataset/Register_faces',
            sklearn_file='Recognition/models/model_sklearn_random_forest.sav'):
    # Detection model
    model_detection = load_model(model_detection_file)
    model_rpn = load_model(model_rpn_file)

    if sklearn_file[-4:] == '.sav':
        sklearn_model = pickle.load(open(sklearn_file, 'rb'))
    else:
        sklearn_model = sklearn_file
        
    # Recognition model
    #model = load_model("Recognition/models/last_facenet_the_feature_extractor.h5")
    if 'facenet_keras' not in model_reco_file:
        model = load_model(model_reco_file)
        faceNet = model.get_layer('model_1')
    else:
        faceNet = load_model(model_reco_file)

    # create dataset
    dataset = load_dataset('pickle_dataset.p')#faceNet, folder=dataset_file)
    treshold = 1

    return model_detection, model_rpn, faceNet, sklearn_model, dataset


def update_dataset(dataset, name, faceNet, faces):
    faces_ = []
    for f in faces:
        if len(f) == 1:
            faces_.append(cv2.resize(f[0], (128,128)))

    embeddings = faceNet.predict(np.array(faces_)/255)
    dataset[name] = embeddings
    pickle.dump(dataset, open('pickle_dataset.p','wb'))
    return dataset



def login(name, image, dataset, treshold, model_detection=None, model_rpn=None, faceNet=None, k=5, show=False):
    if model_detection is None:
        model_detection = load_model('Detection/models/best_detector_inception.h5')
    if model_rpn is None:
        model_rpn = load_model('Detection/RPN/models/best_rpn256_lfw_transfer.h5')
    if faceNet is None:
        model = load_model("Recognition/models/facenet_xception_block10+.h5")
        faceNet = model.get_layer('functional_1')

    m = None
    if show:
        m = 'box'
    faces = detectFaces(image, model_rpn, get_region_RPN, model_detection, show_mode=m)
    if show:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return testFaces(name, faces, dataset, treshold, model_detection=None, model_rpn=None, faceNet=None, k=5, show=False)[0]



def testFaces(name, faces, dataset, treshold, model_detection=None, model_rpn=None, faceNet=None, sklearn_model=None, k=5, show=False):
    if model_detection is None:
        model_detection = load_model('Detection/models/best_detector_inception.h5')
    if model_rpn is None:
        model_rpn = load_model('Detection/RPN/models/best_rpn256_lfw_transfer.h5')
    if sklearn_model is None:
        sklearn_model = pickle.load(open('Recognition/models/model_sklearn_random_forest.sav', 'rb'))
    if faceNet is None:
        model = load_model("Recognition/models/facenet_xception_block10+.h5")
        faceNet = model.get_layer('functional_1')

    if name not in dataset.keys():
        return False,None

    identities = []
    dists = []
    for i,face in enumerate(faces):
        face = cv2.resize(face,(128,128))
        encoded_img = faceNet.predict(np.array([face])/255)

        if sklearn_model == 'euclidian':
            distances = sorted([euclidean_distance([encoded_img, vect]) for vect in dataset[name]])
        elif sklearn_model == 'cosine':
            distances = sorted([distance.cosine(encoded_img, vect) for vect in dataset[name]])
        elif sklearn_model == None:
            distances = sorted([euclidean_distance([encoded_img, vect]) for vect in dataset[name]])
        else :
            X = np.array([encoded_img[0] - vect for vect in dataset[name]])
            distances = sklearn_model.predict_proba(X)
            distances = distances[:,1]

##            f = open('debug.txt','w')
##            f.write(str(distances))
##            f.close()


        distances = distances[:k]
        d = float(sum(distances)/len(distances))

        if show:
            cv2.imshow('face '+str(i),face)
            #print(encoded_img)
            print(dataset[name][0])
            print('Distance : ',round(d,3))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        dists.append(d)
        
        if d < treshold:
            return True, dists
        

    return False, dists



def recognizeFace(image, dataset, encoder, sklearn_model, treshold=.5, k=5):
    """Reconnait le visage d'entrée. img est croppée autour du visage"""
    if len(image) > 0:
        image = image[0]
        image = cv2.resize(image, (128,128) )/255
    else:
        return None
    results = {}
    encoded = encoder.predict(np.array([image]))
    # Predict similarities
    for name in dataset:
        if sklearn_model == 'euclidian' or sklearn_model == None:
            sims = [euclidean_distance([encoded, vect]) for vect in dataset[name]]
        elif sklearn_model == 'cosine':
            sims = [distance.cosine(encoded, vect) for vect in dataset[name]]
        else:
            sims = [sklearn_distance(encoded[0], vect, sklearn_model) for vect in dataset[name]]
        results[name] = sorted(sims)[:k]
        results[name] = float(sum(results[name]) / len(results[name]))


    # Pick lowest dist
    names = [name for name in results]
    similarities = [results[name] for name in results]
    sorted_names = [x for _,x in sorted(zip(similarities,names))]

    return sorted_names





def showFaces(image, model_rpn, model_detection):
    '''test d'affichage des têtes'''
    im = detectFaces(image, model_rpn, get_region_RPN, model_detection, show_mode='online')
    return im


def store_faces_in_file(faces, name, file):
    """Enregistre les images au format nom_1 dans le fichier file"""
    c = 0
    for face in faces:
        if len(face) == 1:
            face = face[0]
            cv2.imwrite(file+'/'+name+'_'+str(c)+'.jpg', face)
            c += 1
    return True
    
def isInvited(identifiant):
    file = 'registered_names.txt'
    f = open(file,'r')
    for name in f:
        if name==identifiant+'\n':
            return True
    return True #set False if you want the person who tries to create account to be invited in the txt file, easier to test without this option

def delete_invitation(to_delete):
    file = 'registered_names.txt'
    f = open(file,'r')
    names = f.readlines()
    f.close()
    print(names)
    f = open("registered_names.txt","w")
    for name in names:
      if name!=to_delete+"\n":
        f.write(name)
    f.close()

if __name__ == '__main__':


    # /!\ A NE METTRE QU'EN DEBUT DE PROGRAMME (chargement des variables globales)
    DATASET_FILE = 'Recognition/dataset/Register_faces'
    DATASET = []
    model = load_model('Recognition/models/last_facenet_simple_encoder_bis_transfer1.h5')
    encoder = model.get_layer('model_1')
    simulateEntryFile(encoder, folder=DATASET_FILE)
##    model_detection, model_rpn, faceNet, sklearn_model, DATASET = loadVar(model_detection_file='Detection/models/best_detector_inception.h5',
##                                                           model_rpn_file='Detection/RPN/models/last_RPN_workstation_output.h5',
##                                                           model_reco_file="Recognition/models/last_facenet_simple_encoder_bis_transfer1.h5",
##                                                           #sklearn_file='Recognition/models/model_sklearn_random_forest.sav',
##                                                           sklearn_file='cosine',
##                                                           dataset_file=DATASET_FILE)
    treshold = 1


##    file = 'C:/Users/Pirlouit/source/python/MachineLearning/projectISEN/RASSEMBLEMENT/Login_v1/Recognition/dataset/Register_faces'
##    pics_name = os.listdir(file)        
##    random.shuffle(pics_name)
##    for f in pics_name:
##        im = cv2.imread(file+'/'+f)
##
##        #cv2.imshow('im',im)
##        #print(encoded_img)
##        #print(dataset[name][0])
##        #print('Distance : ',round(d,3))
##        #cv2.waitKey(0)
##        #cv2.destroyAllWindows()
##
##        
##        
##        # LA FONCTION DE LOGIN A INSERER (Renvoie True ou False)
##        # Mettre show=False sur le site (debugage uniquement)
##        log, dist = testFaces('guigui', [im],
##                    DATASET, treshold,
##                    model_detection=model_detection,
##                    model_rpn=model_rpn,
##                    faceNet=faceNet,
##                    sklearn_model=sklearn_model,
##                    k=5, show=True)
##
##
##        
##        print('distance : ',dist)


        

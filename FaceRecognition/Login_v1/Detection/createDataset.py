#################################################################################################################################
#                                                                                                                               #
#                                                              IMPORTS                                                          #
#                                                                                                                               #
#################################################################################################################################
import cv2
import numpy as np
import pyautogui
import time
from cropImages import *
import os
import random

#################################################################################################################################
#                                                                                                                               #
#                                                              FONCTIONS                                                        #
#                                                                                                                               #
#################################################################################################################################
def getFaces(image, model, show_mode='crop'):
    # extract the dimensions , Resize image into 300x300 and converting image into blobFromImage
    (h, w) = image.shape[:2]

    # blobImage convert RGB (104.0, 177.0, 123.0)
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()
    faces = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY: endY, startX: endX]
            if 0 not in face.shape:
                faces.append([face,startX,startY,endX,endY])

    return faces


def takePictures(folder='dataset/raw_pics/',cap=None,starting_index=1):
      while(1):
        ret, frame = cap.read()
        cv2.imshow('webcam',frame)
        k = cv2.waitKey(5) & 0xFF
        if (starting_index%5==0):
            cv2.imwrite(folder+'raw_pic_'+str(time.time())+'.jpg',frame)
        starting_index+=1

        
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA+1) * max(0, yB- yA +1 )
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou    


            
def classifySubPic(sub_pic, faces, treshold):
    if len(faces)>1 or len(faces)==0:
        return None
    crop_img ,startX, startY, endX, endY = sub_pic
    crop_face ,startX_face, startY_face, endX_face, endY_face = faces[0]
    if iou([startX,startY,endX,endY], [startX_face, startY_face, endX_face, endY_face]) > treshold:
        return 'face'
    else:
        return 'not_face'
    

def loadModel(deploy_proto='CAFFE_DNN/deploy.prototxt.txt',
              caffemodel='CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel'):
    model = cv2.dnn.readNetFromCaffe(deploy_proto, caffemodel)
    return model

def createDataset(folder='dataset/raw_pics', folder_f='dataset/face', folder_nf='dataset/not_face', treshold=0.5):
    model = loadModel()
    i = 0
    for file in os.listdir(folder):
        img = cv2.imread(folder+'/'+file)
        faces = getFaces(img, model, show_mode=None)
        sub_pics = slidingWindows(img, 150, 190, 25, 25)   
        for pic in sub_pics:
            i += 1
            if classifySubPic(pic, faces, treshold) =='face':
                cv2.imwrite(folder_f+'/'+str(time.time())+'.jpg',pic[0])
            elif classifySubPic(pic, faces, treshold) =='not_face':
                if i%15 == 0:
                    cv2.imwrite(folder_nf+'/'+str(time.time())+'.jpg',pic[0])
                i += 1
    return 
    

                
                
        
def getXY_detection(folder_head, folder_not_head, size, prop=0.5, imsize=128):
    heads = [cv2.resize(cv2.imread(folder_head+'/'+f), (imsize, imsize)) for f in os.listdir(folder_head)]
    not_heads = [cv2.resize(cv2.imread(folder_not_head+'/'+f), (imsize, imsize)) for f in os.listdir(folder_not_head)]

    random.shuffle(heads)
    random.shuffle(not_heads)

    X = []
    y = []
    for i in range(int(size*prop)):
        X.append(heads[i])
        y.append([1,0])
        
    for i in range(int(size*(1-prop))):
        X.append(not_heads[i])
        y.append([0,1])

    temp = list(zip(X, y))
    random.shuffle(temp)
    X, y = zip(*temp)

    X, y = np.array(X)/255.0,np.array(y)
    return X, y
        
    
    
    
    
    
        
#################################################################################################################################
#                                                                                                                               #
#                                                              MAIN                                                             #
#                                                                                                                               #
#################################################################################################################################
if __name__ == '__main__':
    ### Camera settings
    SCREEN_SIZE = pyautogui.size()
    WEBCAM_SIZE = (1000, 600)

    #Open Camera object
    cap = cv2.VideoCapture(0)

    #Decrease frame size
    cap.set(cv2.cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_SIZE[0])
    cap.set(cv2.cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_SIZE[1])

    takePictures(folder='dataset/raw_pics/',cap=cap)
##    print('create dataset')
##    createDataset(folder='raw_pics', folder_f='face', folder_nf='not_face', treshold=0.85)

      






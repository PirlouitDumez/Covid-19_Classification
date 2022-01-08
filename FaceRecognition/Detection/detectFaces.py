#################################################################################################################################
#                                                                                                                               #
#                                                              IMPORTS                                                          #
#                                                                                                                               #
#################################################################################################################################
import os
import time
os.environ['KERAS_BACKEND'] = 'tensorflow'
import warnings
warnings.simplefilter("ignore")

import cv2
import numpy as np
import random


from keras.models import load_model

#################################################################################################################################
#                                                                                                                               #
#                                                          FONCTIONS                                                            #
#                                                                                                                               #
#################################################################################################################################
def non_max_suppress(predicts_dict, threshold=0.2):
    """
    implement non-maximum supression on predict bounding boxes.
    Args:
        predicts_dict: {"stick": [[x1, y1, x2, y2, scores1], [...]]}.
        threshhold: iou threshold
    Return:
        predicts_dict processed by non-maximum suppression
    """
    for object_name, bbox in predicts_dict.items():   #NMS for each category of objectives
        bbox_array = np.array(bbox, dtype=np.float)

        ## Get the coordinates and confidence of all bounding box es (bbx) under the current target category, and calculate the area of all BBX
        x1, y1, x2, y2, scores = bbox_array[:,0], bbox_array[:,1], bbox_array[:,2], bbox_array[:,3], bbox_array[:,4]
        areas = (x2-x1+1) * (y2-y1+1)
        #print("areas shape = ", areas.shape)

        ## Sort all bbx confidence s under the current category from high to low (order saves index information)
        order = scores.argsort()[::-1]
        print("order = ", order)
        keep = [] #Index information used to store the final retained bbx

        ## Traverse bbx from high to low by confidence, removing all rectangular boxes with IOU values greater than threshold
        while order.size > 0:
            i = order[0]
            keep.append(i) #Keep the bbx index corresponding to the current maximum confidence

            ## Get all the upper-left and lower-right coordinates corresponding to the intersection of the current bbx, and compute the IOU (note that here is the IOU of one BBX and all other BBX at the same time)
            xx1 = np.maximum(x1[i], x1[order[1:]])  #When order.size=1, the following results are np.array([]), which does not affect the final results.
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2-xx1+1) * np.maximum(0.0, yy2-yy1+1)
            iou = inter/(areas[i]+areas[order[1:]]-inter)
            print("iou =", iou)

            print(np.where(iou<=threshold)) #Output bbx index that has not been removed (index relative to iou vector)
            indexs = np.where(iou<=threshold)[0] + 1 #Get the retained index (since the IOU is not computed, the index differs by 1 and needs to be added)
            print("indexs = ", type(indexs))
            order = order[indexs] #Update the retained index
            print("order = ", order)
        bbox = bbox_array[keep]
        predicts_dict[object_name] = bbox.tolist()
        predicts_dict = predicts_dict
    return predicts_dict




def detectFaces(image, model_rpn, get_region_RPN, model_detection, show_mode='box'):
    """Extract faces from image"""
    faces = []
    boxes = []
    image_red = np.copy(image)

    # Crop image
    sub_pics = get_region_RPN(image, model_rpn, imsize=256, treshold=0.5, min_area=1000, out_imsize=128)
    boxes = []

    if model_detection is None:
        for i,s in enumerate(sub_pics):
            (startX, startY, endX, endY) = np.array(sub_pics[i][1:]).astype("int")
            boxes.append([startX, startY, endX, endY, 1])

##    #sub_pics = slidingWindows(image,150,190,50,35)

    else:
        p = [cv2.resize(pic[0], (128,128))/255.0 for pic in sub_pics]


        # Make predictions
        preds = []
        if len(p) >= 1:
            preds = model_detection.predict(np.array(p))

        # Loop over predictions
        max_pred = 0
        max_box = None#(0,0,1,1)
        for i,pred in enumerate(preds):
            classe = np.argmax(pred)
            proba = list(pred)[classe]

            if classe == 0:
                (startX, startY, endX, endY) = np.array(sub_pics[i][1:]).astype("int")

                boxes.append([startX, startY, endX, endY, proba])




##    boxes = non_max_suppress({"face": boxes}, 0.2)
##    boxes = boxes['face']


    # Mise à l'echelle et affichage des boxes
    s = image.shape
    imsize = (256,256)
    for box in boxes:
        (startX, startY, endX, endY, proba) = box
        startX = int(startX*(s[1]/imsize[1]))
        startY = int(startY*(s[0]/imsize[0]))
        endX = int(endX*(s[1]/imsize[1]))
        endY = int(endY*(s[0]/imsize[0]))

        faces.append(image[startY: endY, startX: endX])

        
        text = "{:.2f}%".format(proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image_red, (startX, startY), (endX, endY),(0, 0, 255), 2)
        #cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


    if len(faces) > 0 and show_mode == 'crop':
        image = cv2.imshow('out', faces[0])
    elif show_mode == 'box':
        cv2.imshow('out', image_red)
    elif show_mode == 'online':
        return image_red, faces
    return faces



#################################################################################################################################
#                                                                                                                               #
#                                                              MAIN                                                             #
#                                                                                                                               #
#################################################################################################################################

##if __name__ == '__main__':
##    ### Camera settings
##    SCREEN_SIZE = pyautogui.size()
##    WEBCAM_SIZE = (1000, 600)
##
##    # Open Camera object
##    cap = cv2.VideoCapture(0)
##
##    # Decrease frame size
##    cap.set(cv2.cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_SIZE[0])
##    cap.set(cv2.cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_SIZE[1])
##
##    ### Model settings
##    # Reads the network model stored in Caffe framework's format.
##    print("Loading models...................")
##    model_detection = loadModel('models/best_detector_inception.h5')
##    model_rpn = load_rpn_model((256, 256, 3), 'RPN/models/best_rpn256_lfw_transfer.h5')
##
##    while 1:
##        ret, frame = cap.read()
##
##        t1 = time.time()
##        faces = detectFaces(frame, model_rpn, model_detection, show_mode='box')
##        print(time.time() - t1)
##
##        k = cv2.waitKey(5) & 0xFF
##


if __name__ == '__main__':
    from cropImages import *

    model_detection = load_model('models/best_detector_inception.h5')
    model_rpn = load_model('RPN/models/best_rpn256_lfw_transfer.h5')
    file = 'dataset/raw_pics_test'
    pics_name = os.listdir(file)
    random.shuffle(pics_name)
    for f in pics_name:
        image = cv2.imread(file+'/'+f)
        detectFaces(image, model_rpn, model_detection, show_mode='box')

        cv2.waitKey(0)
        cv2.destroyAllWindows()










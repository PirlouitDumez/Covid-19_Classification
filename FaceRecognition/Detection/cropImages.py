#################################################################################################################################
#                                                                                                                               #
#                                                              IMPORTS                                                          #
#                                                                                                                               #
#################################################################################################################################
import cv2
import numpy as np
import pyautogui
import random
from keras.models import load_model
import os



#################################################################################################################################
#                                                                                                                               #
#                                                        SLIDING WINDOWS                                                        #
#                                                                                                                               #
#################################################################################################################################
def slidingWindows(img,x_size,y_size,x_pad,y_pad):
    sub_pics = []
    s = img.shape
    print(img.shape)
    for i in range(int((s[1]-x_size)/x_pad)):
        x = x_pad*i
        for j in range(int((s[0]-y_size)/y_pad)):
            y = y_pad*j
            crop_img = img[y:min(s[0], y+y_size), x:min(s[1],x+x_size)]
            sub_pics.append([crop_img,x,y,x_size+x,y_size+y])
    return sub_pics 


#################################################################################################################################
#                                                                                                                               #
#                                                               RPN                                                             #
#                                                                                                                               #
#################################################################################################################################
def get_region_RPN(image, model, imsize=256, treshold=0.5, min_area=500, out_imsize=128):
    """return crop_image, x1, y1, x2, y2 using RPN moddel"""
    image = cv2.resize(image, (imsize, imsize))
    contours = get_proposed_regions(image, model, imsize=imsize, treshold=treshold, min_area=min_area)
    crop_images = []
    for contour in contours:
        (x1,y1),(x2,y2) = contour
        crop_image = image[y1:y2,x1:x2]
        crop_image = cv2.resize(crop_image, (out_imsize,out_imsize))
        crop_images.append([crop_image, x1, y1, x2, y2])
##        cv2.imshow('crop',crop_image)
##        cv2.waitKey()
##        cv2.destroyAllWindows()
    return crop_images





def seuil(im, treshold=0.5):
    """Grayscale to Black and white"""
    im[im>treshold] = 1
    im[im<=treshold] = 0
    return im
    

def get_contours(im, min_area=500):
    """Find whiter rectangles on image"""
    #print(1)
    edged=cv2.Canny(im,0,255)
    print("")
    
    contours, hierarchy=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    points = []
    for c in contours:
        xmin = min(list(c[:,0,0]))
        ymin = min(list(c[:,0,1]))
        xmax = max(list(c[:,0,0]))
        ymax = max(list(c[:,0,1]))

        if (xmax-xmin)*(ymax-ymin) >= min_area:
            start = (xmin,ymin)
            end = (xmax, ymax)
            points.append([start, end])

    return points



def get_proposed_regions(image, model, imsize=512, treshold=.5, min_area=500):
    """Use RPN model to propose regions of interest"""

    # Predict areas of interest
    image = cv2.resize(image, (imsize, imsize))
    pic = model.predict(np.array([image/255]))[0]

    # Traite les images de sortie
    pic = seuil(pic, treshold=treshold)
    pic = (255*pic).astype('uint8')
    contours = get_contours(pic, min_area=min_area)

    return contours
    


def test_rpn_model(image, model, imsize=512, treshold=.5, min_area=500):
    cv2.imshow('input',image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # Predict areas of interest
    image = cv2.resize(image, (imsize, imsize))
    pic = model.predict(np.array([image/255]))[0]

    # Traite les images de sortie
    cv2.imshow('output',pic)
    cv2.waitKey()
    cv2.destroyAllWindows()

    for i in range(1, 10):
        #pic_s = (255*pic).astype('uint8')
        #contours = get_contours(pic_s, min_area=min_area)
        print(np.min(pic),np.max(pic))
        cv2.imshow('output',seuil(pic, treshold=i/10))
        print("seuil : ",i/10)
        cv2.waitKey()
        cv2.destroyAllWindows()
    

#################################################################################################################################
#                                                                                                                               #
#                                                              MAIN                                                             #
#                                                                                                                               #
#################################################################################################################################
if __name__ == '__main__':
    model = load_model('RPN/models/last_RPN_workstation_output.h5')
    file = 'dataset/raw_pics'
    pics = [cv2.imread(file+'/'+f) for f in os.listdir(file)]
    random.shuffle(pics)

    for p in pics:
        test_rpn_model(p, model, imsize=256, treshold=0.5, min_area=1000)
##        cropped = get_region_RPN(p, model, imsize=256, treshold=0.5, min_area=1000, out_imsize=128)
##        cv2.imshow('init', p)
##        for i,c in enumerate(cropped):
##            cv2.imshow('crop_'+str(i),cv2.resize(c[0], (128,128)))
##        cv2.waitKey()
##        cv2.destroyAllWindows()











    
    

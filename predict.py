# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 08:30:25 2021

@author: giuli
"""

import os
import cv2
import pickle
from tensorflow.keras.models import load_model

def main():
    
    cwd=os.getcwd()
    test=os.path.join(cwd,"test")
    
    # load the model and label binarizer
    netname = 'simplenn'
    netinfo  = {'simplenn':{"pickle":"simple_nn_lb.pickle",'model':'simple_nn.model'  ,'size':32,'flatten':True},
                'vggnet'  :{"pickle":"smallvggnet.pickle" ,'model':'smallvggnet.model','size':64,'flatten':False}}
   
    
    print("[INFO] loading network and label binarizer...")
    model = load_model(os.path.join(cwd,"output",netinfo[netname]["model"]))
    lb = pickle.loads(open(os.path.join(cwd,"output",netinfo[netname]["pickle"]), "rb").read())
    
    for f in os.listdir(test):
        
        print("[INFO] Reading :"+f)
        
        image=cv2.imread(os.path.join(test,f))
        output=image.copy()
        
        #resize to model dimensions
        image = cv2.resize(image, (netinfo[netname]["size"], netinfo[netname]["size"]))
        
        # scale the pixel values to [0, 1]
        image = image.astype("float") / 255.0   
        
        # check to see if we should flatten the image and add a batch
        # dimension
        
        if netinfo[netname]["flatten"]:
            image = image.flatten()
            image = image.reshape((1, image.shape[0]))
            # otherwise, we must be working with a CNN -- don't flatten the
            # image, simply add the batch dimension
        else:
            image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))
        
        # make a prediction on the image
        preds = model.predict(image)
        
        # find the class label index with the largest corresponding
        # probability
        i = preds.argmax(axis=1)[0]
        label = lb[i]
        print("[INFO] prediction {}: ".format(preds))
        
        # draw the class label + probability on the output image
        text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
        cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
        
        # show the output image
        cv2.imshow("Image", output)
        cv2.waitKey(0)
        
        
if __name__ == "__main__":
   main()
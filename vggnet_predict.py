# import the necessary packages
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

import numpy as np
import pickle
import cv2
import os


def read_and_preprocess(path):
  image = load_img(path, target_size=(224, 224))
  image = img_to_array(image)
  image = np.expand_dims(image, axis=0)
  image = preprocess_input(image)
  return image


def main():
    # load the VGG16 network and initialize the label encoder
    print("[INFO] loading network...")
    model = VGG16(weights="imagenet", include_top=False)
    
    # load the model and label binarizer
    cwd=os.getcwd()
    test=os.path.join(cwd,"test")
    
    lb     = pickle.loads(open(os.path.join(cwd,"output","le.cpickle"), "rb").read())
    logreg = pickle.loads(open(os.path.join(cwd,"output","vggnet_transfer_model.cpickle"), "rb").read())
    
    for f in os.listdir(test):
        
        print("[INFO] Reading :"+f)
        output=cv2.imread(os.path.join(test,f))
        
        # preprocess image for feature extraction
        image = read_and_preprocess(os.path.join(test,f))
        
        # extract features
        features = model.predict(image, batch_size=1)
        features = features.reshape((features.shape[0], 7*7*512))
        
        # predict classes
        preds = logreg.predict_proba(features)
        
        # find the class label index with the largest corresponding
        # probability
        i = preds.argmax(axis=1)[0]
        label = lb.classes_[i]
        print("[INFO] prediction {}: ".format(preds))
        
        # draw the class label + probability on the output image
        text = "{}: {:.2f}%".format(label, preds[0][i] * 100)
        cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
        
        # show the output image
        cv2.imshow("Image", output)
        cv2.waitKey(0)
    
    return

if __name__ == "__main__":
   main()
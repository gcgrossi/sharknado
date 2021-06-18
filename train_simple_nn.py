# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 10:28:01 2021

@author: giuli
"""


#import tensorflow as tf
#from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import random
import os


file_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
keep_labels     = ['great_white_shark','hammerhead_shark']

def list_files(indir=os.getcwd(),valid_extensions=file_extensions,valid_labels=keep_labels):
    for (rootdir,dirs,files) in os.walk(indir):
        for filename in files:
            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()
            
            # check to see if the file is an image and should be processed
            if valid_extensions is None or ext.endswith(valid_extensions):
                
                # construct the path to the image and yield it
                imagePath = os.path.join(rootdir, filename)
                
                # yield the path if the label should not be dropped 
                if imagePath.split(os.path.sep)[-2] in valid_labels:
                    yield imagePath
            
    return

def main():
    
    dataset_path = os.path.join(os.getcwd(),'sharks')
    output_path= os.path.join(os.getcwd(),"output")
    
    #obtain image paths and ramdomize it
    image_paths = list(list_files(dataset_path))
    random.seed(42)
    random.shuffle(image_paths)
    
    # initialize data and labels list
    data, labels, count = [],[],0
    
    print('[INFO] reading images from disk and resize ... this may take a while')
    for i in image_paths:
        # load the image, resize the image to be 32x32 pixels (ignoring
        # aspect ratio), flatten the image into 32x32x3=3072 pixel image
    	# into a list, and store the image in the data list

        image = cv2.imread(i)
        image = cv2.resize(image, (32, 32)).flatten()
        data.append(image)
       
        label = i.split(os.path.sep)[-2]
        labels.append(label)
        
        count+=1
        if count==-1: break

    # print label count
    label_list = os.listdir(dataset_path)
    for l in label_list: print("label: {} counts: {}".format(l,labels.count(l)))

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data,dtype="float") / 255.0
    labels = np.array(labels)
    
    # partition the data into training and testing splits using 75% of
    # the data for training and the remaining 25% for testing
    (trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.3, random_state=42)
    
    original_classes = trainY
    
    # integer encode
    label_encoder = LabelEncoder()
    trainY = label_encoder.fit_transform(trainY)
    testY  = label_encoder.transform(testY)
    
    print(original_classes[0:10])
    print(trainY[0:10])
    u, indices =np.unique(trainY,return_index=True)
    classes = [original_classes[i] for i in indices]
    print(classes)
  
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    trainY = trainY.reshape(len(trainY), 1)
    trainY = onehot_encoder.fit_transform(trainY)
    testY = testY.reshape(len(testY), 1)
    testY = onehot_encoder.transform(testY)
    print(trainY[0:10])
    

    # define the 3072-1024-512-3 architecture using Keras
    model = Sequential()
    model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
    model.add(Dense(512, activation="sigmoid"))
    model.add(Dense(len(trainY[0]), activation="softmax"))

    # initialize our initial learning rate and # of epochs to train for
    INIT_LR = 0.03
    EPOCHS = 140
    # compile the model using SGD as our optimizer and categorical
    # cross-entropy loss (you'll want to use binary_crossentropy
    # for 2-class classification)
    print("[INFO] training network...")
    opt = SGD(learning_rate=INIT_LR)
    #opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
    
    # train the neural network
    H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=64)
    
    # evaluate the network
    print("[INFO] evaluating network...")
    predictions = model.predict(x=testX, batch_size=64)
    print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=classes))
   #print(confusion_matrix(testY.argmax(axis=1),predictions.argmax(axis=1), labels=classes))
    
    # plot the training loss and accuracy
    N = np.arange(0, EPOCHS)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["accuracy"], label="train_acc")
    plt.plot(N, H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy (Simple NN)")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(output_path,'simple_nn_plot.png'))
    
    # save the model and label binarizer to disk
    print("[INFO] serializing network and label binarizer...")
    model.save(os.path.join(output_path,'simple_nn.model'), save_format="h5")
    f = open(os.path.join(output_path,'simple_nn_lb.pickle'), "wb")
    f.write(pickle.dumps(classes))
    f.close()


    return


if __name__ == "__main__":
    main()
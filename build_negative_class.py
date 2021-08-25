import os
import pickle
import cv2
import random

from google_vision.gvision import *

import numpy as np
from tensorflow.keras.models import load_model
from imutils.object_detection import non_max_suppression


def selective_search(image, method="fast"):

	# initialize OpenCV's selective search implementation and set the
	# input image
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    # check to see if we are using the *fast* but *less accurate* version
    # of selective search
    if method == "fast":
	    ss.switchToSelectiveSearchFast()
    # otherwise we are using the *slower* but *more accurate* version
    else:
	    ss.switchToSelectiveSearchQuality()
    # run selective search on the input image
    rects = ss.process()
    
    # return the region proposal bounding boxes
    return rects

def compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


file_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
keep_labels     = ['great_white_shark','hammerhead_shark','whale_shark']

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
    
    # speed-up using multithreads
    cv2.setUseOptimized(True)   
    cv2.setNumThreads(4)

    cwd=os.getcwd()
    dataset_path=os.path.join(cwd,"sharks")

    # initialize google vision api
    gvision_auth=os.path.join(cwd,"google_vision","gvision_auth.json")
    gvis = GVisionAPI(gvision_auth)
    
    # load the model and label binarizer
    netname = 'vggnet_finetune'
    netinfo  = {'simplenn'         :{"pickle":"simple_nn_lb.pickle",'model':'simple_nn.model'  ,'size':32,'flatten':True},
                'vggnet'           :{"pickle":"smallvggnet.pickle" ,'model':'smallvggnet.model','size':64,'flatten':False},
                'vggnet_finetune'  :{"pickle":"vggnet_finetune.pickle" ,'model':'vggnet_finetune.model','size':224,'flatten':False}}
   

    #obtain image paths and ramdomize it
    image_paths = list(list_files(dataset_path))
    random.seed(42)
    random.shuffle(image_paths)

    max_negative = 1000
    negative_count = 0
    image_count = 0    

    print("[INFO] Reading images from disk. This may take a while ... ")    
    for i in image_paths:
        image_count+=1

        #read the image
        image = cv2.imread(i)
        
        #copy image for drawing
        clone = image.copy()

        # reshape image maintaining aspect ratio
        # should speed-up selective search
        (H,W) = image.shape[:2]
        newH = 200
        newW = int(W*200/H)
        image = cv2.resize(image, (newW, newH))

        image_area = image.shape[0]*image.shape[1]

        #perform a request to the API
        gvis.perform_request(image,'object detection')
        headers,objs = gvis.objects()

        # check if image contains a shark
        shark_detected = 0
        for obj in objs: 
            if obj[0] in ['Animal','Shark']:
                true_box = (obj[2][0],obj[2][1],obj[4][0],obj[4][1])
                shark_detected+= 1

                # draw the true rectangle
                cv2.rectangle(image,(int(obj[2][0]),int(obj[2][1])),(int(obj[4][0]),int(obj[4][1])),(0,255,0), 2)
                cv2.imshow("tue", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        if shark_detected != 1: continue

        rects = selective_search(image)
        # loop over the region proposal bounding box 
        for rect in rects:
         
            # initialize roi
            roi = None

            # unpack the proposed/true rectangle bounding box
            (propStartX, propStartY, propEndX, propEndY) = rect
            (gtStartX, gtStartY, gtEndX, gtEndY) = true_box
            
            iou = compute_iou(true_box,rect)

            # determine if the proposed bounding box falls *within*
			# the ground-truth bounding box
            fullOverlap = propStartX >= gtStartX
            fullOverlap = fullOverlap and propStartY >= gtStartY
            fullOverlap = fullOverlap and propEndX <= gtEndX
            fullOverlap = fullOverlap and propEndY <= gtEndY

            # check to see if there is not full overlap *and* the IoU
			# is less than 20% *and* we have not hit our negative
			# count limit
            #not fullOverlap and
            if iou < 0.2 and negative_count <= max_negative:
				# extract the ROI and then derive the output path to
				# the negative instance
                roi = image[propStartY:propEndY, propStartX:propEndX]
                roi_area = roi.shape[0]*roi.shape[1]
                
                # skip if roi area too small
                if roi_area/image_area < 0.05 : continue

                # resize roi to CNN input dimensions
                roi = cv2.resize(roi, (netinfo[netname]['size'], netinfo[netname]['size']))
                
                #draw rois
                c = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                cv2.rectangle(clone,(propStartX, propStartY),(propEndX, propEndY),c, 2)
            

                # write roi to disk
                filename = "{}.png".format(negative_count)
                outputPath = os.path.join(dataset_path,"not_shark",filename)
                #cv2.imwrite(outputPath, roi)
                
                negative_count+=1
                if negative_count>=max_negative: break
        
        cv2.imshow("rois", clone)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if negative_count>=max_negative: break


    return

    
       
        
if __name__ == "__main__":
   main()
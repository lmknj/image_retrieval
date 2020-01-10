import os
import sys
import time
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

for directory in os.listdir(MODEL_DIR):
    tmp = os.path.join(MODEL_DIR, directory)
    if len(os.listdir(tmp)) == 0:
        os.rmdir(tmp)

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "model_data/mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

while True:
    # Read user input
    file_name = input("Enter an image name or type 'quit': ")

    if file_name == 'quit':
        break

    if not os.path.exists(file_name):
        print("Error: '{}' does not exist.".format(file_name))
        continue

    # Read an image
    image = skimage.io.imread(file_name)
    #image = cv2.imread(file_name)
    


    """
    #透過度ミス
    image2 = image
    for x in range(len(image2[0])):
        for y in range(len(image2)):
                image2[y][x] = [image2[y][x][0],
                                image2[y][x][1],
                                image2[y][x][2],
                                int(128)]                                ]
    
    print(image2)
    plt.imshow(image2)
    plt.show()
    """

    
    
    # Run detection
    
    start = time.time()
    results = model.detect([image], verbose=1)
    end = time.time()
    print("Detection Time: {:.3f} sec".format(end - start))
    
    # Visualize results
    r = results[0]
    


    mask = r["masks"]
    classID =  r['class_ids']
    
    """
    print("mask", mask)
    print("mask[0][0]",type(mask[0][0]))
    print("mask[0][0][0]",type(mask[0][0][0]))
    print("mask[0](横)",mask[0],"len", len(mask[0]))
    print("mask(縦)",mask,"len", len(mask))
    print("mask[0][0](各要素)",mask[0][0],"len", len(mask[0][0]))
    """
    print("classid", classID)
    print("classid[0]", classID[0])
    
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'], ax=None)
    
    
    #print("image[0](横)",image[0],"len", len(image[0]))
    #print("image(縦)",image,"len", len(image))
    #print("image[0][0](各要素)",image[0][0],"len", len(image[0][0]))
    
    
    import copy
    image2 = copy.deepcopy(image)  
    
    #for m in range(len(classID)):    
    for x in range(len(image2[0])):
        for y in range(len(image2)):
            if True in mask[y][x]:
                isMask = False
                for m in range(len(classID)):  
                    if (mask[y][x][m] == True) and (classID[m] == 1):
                        image2[y][x] = [255,255,255]
                        isMask = True
                        break
                    
                if not isMask:
                    image2[y][x] = [0,0,0]
            else:
                image2[y][x] = [0,0,0]

            """         
            if True in mask[y][x] & mask[y][x][m] == True &(classID[m] == 1): #classID配列の要素が１になるとき
                    image2[y][x] = [255,255,255]
            elif True in mask[y][x] & (classID[m] != 1): #classID配列の要素が１になるとき
                continue
            else:
                image2[y][x] = [0,0,0]
            """
            """
            if mask[y][x][m] == True:
                image2[y][x] = [255,255,255]#する#何とか透過させたい
            if mask[y][x][m] == False:
                image2[y][x] = [0,0,0]#黒にする#何とか透過させたい
            """

    """
    cv2.imwrite("output1.png",image2)
    cv2.imshow("procecced",image2)
    

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    #plt.imshow(image2)
    #plt.show()
    from skimage import viewer
    new_viewer = viewer.ImageViewer(image2) 
    new_viewer.show() 
    










    







    
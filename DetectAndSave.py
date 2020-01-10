import os
import sys
import time
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import pickle



from mrcnn2 import utils
import mrcnn2.model as modellib
from mrcnn2 import visualize

# Import COCO config
import coco


class ImageData:      #pickleするオブジェクト
    def __init__(self, image, roi, classID, score, file_name):
        self.image = image
        self.roi = roi
        self.classID = classID
        self.score = score
        
        self.file_name = file_name
        self.sizeX = len(image[0])
        self.sizeY = len(image)
        
        """
        print("image[0](横)",image[0],"len", len(image[0]))
        print("image(縦)",image,"len", len(image))
        print("image[0][0](各要素)",image[0][0],"len", len(image[0][0]))
        """

        
        return


class DetectAndSave:
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

    def __init__(self):
        self.detect()
        return

    def detect(self):
                 # Root directory of the project
        ROOT_DIR = os.path.abspath("./")

        # Import Mask RCNN
        sys.path.append(ROOT_DIR)  # To find local version of the library
  

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
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load weights trained on MS-COCO
        self.model.load_weights(COCO_MODEL_PATH, by_name=True)

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        #os.chdir('./images')    #imagesフォルダに移動
        return 
    

    def post_detect(self, file_name, is_visualize):
        """
        # Read user input
        while True:
            file_name = input("Enter an image name: ")

            if not os.path.exists(file_name):
                print("Error: '{}' does not exist.".format(file_name))
                continue

            break
        """
        # Read an image
        #image = skimage.io.imread("./images/{}".format(file_name))
        image = skimage.io.imread(file_name)
   
        # Run detection
            
        #start = time.time()
        results = self.model.detect([image], verbose=1)
        #end = time.time()
        #print("Detection Time: {:.3f} sec".format(end - start))
            
        # Visualize results
        r = results[0]  #複数枚入力できるうちの1枚目の結果
        roi = r["rois"]       #矩形領域の配列の配列(2次元配列)
        classID =  r['class_ids']
        score = r["scores"]
        """
        print("roiType",type(roi))
        print("IDType",type(classID))
        print("scoreType",type(score))
        print("classid", classID)
        print("classid[0]", classID[0])
        print("rois",roi) 
        print("image[0](横)",image[0],"len", len(image[0]))
        print("image(縦)",image,"len", len(image))
        print("image[0][0](各要素)",image[0][0],"len", len(image[0][0]))
        """

        if is_visualize:
            visualize.display_instances(image, r['rois'], r['class_ids'], 
                                    self.class_names, r['scores'], ax=None)
         

        return image, roi, classID, score, file_name


    def detect_save(self, file_name, is_visualize):
        os.chdir('./images')
        image, roi, classID, score, file_name = self.post_detect(file_name, is_visualize)

        if classID.shape[0] == 0:   #なにも検出されなかったら
          
            os.remove(file_name)


            os.chdir('../')

            return 

        
        imageData = ImageData(image = image,
                        roi = roi,
                        classID = classID, 
                        score = score,
                        file_name = file_name
                        )
        


                
        os.chdir('../pickles')    #picklesフォルダに移動
        with open(file_name + ".pickle", 'wb') as f:
            pickle.dump(imageData, f)
        os.chdir('../')

        print("saveEnd")

        return 

    def while_save(self, is_visualize = True):
        import glob
        os.chdir('./images')
        file_names = glob.glob("*.jpg")
        os.chdir('../')
        for file_name in file_names:
            self.detect_save(file_name, is_visualize)


        return


   

def main():
    d_s = DetectAndSave()
    #d_s.detect_save("kouzudata.pickle", is_visualize = True)
    #d_s.while_save(is_visualize = False)
    d_s.detect_save("aaa.jpg", is_visualize = True)
  


    return


if __name__ == "__main__" :
    main()


import os
import numpy as np
import coco
from keypoint_detection import model as modellib
from keypoint_detection import visualize
from keypoint_detection.model import log
import cv2
import time
import pickle

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

class Keypoint_result:
    def __init__(self, image, roi, keypoints, class_ids,scores,file_name, skeleton):
        #(self, image, roi, classID, score, file_name

        #image,boxes,keypoints,masks,class_ids,scores,class_names,skeleton = inference_config.LIMBS
        #frame,r['rois'],r['keypoints'],r['masks'],r['class_ids'],r['scores'],class_names


        return



ROOT_DIR = os.getcwd()
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "mylogs")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco_humanpose.h5")
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    KEYPOINT_MASK_POOL_SIZE = 7

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights

model_path = os.path.join(ROOT_DIR, "mask_rcnn_coco_humanpose.h5")
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

#class_names = ['BG', 'person']
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
def cv2_display_keypoint(image,boxes,keypoints,class_ids,scores,class_names,skeleton = inference_config.LIMBS):
    # Number of persons
    N = boxes.shape[0]
    if not N:
        print("\n*** No persons to display *** \n")
    else:
        assert N == keypoints.shape[0] and N == class_ids.shape[0] and N==scores.shape[0],\
            "shape must match: boxes,keypoints,class_ids, scores"
    colors = visualize.random_colors(N)
    for i in range(N):
        color = colors[1]
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        for Joint in keypoints[i]:
            if (Joint[2] != 0):
                cv2.circle(image,(Joint[0], Joint[1]), 2, color, -1)

        #draw skeleton connection
        limb_colors = [[0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0], [170, 255, 0],
                       [255, 170, 0], [255, 0, 0], [255, 0, 170], [170, 0, 255], [170, 170, 0], [170, 0, 170]]
        if (len(skeleton)):
            skeleton = np.reshape(skeleton, (-1, 2))
            neck = np.array((keypoints[i, 5, :] + keypoints[i, 6, :]) / 2).astype(int)
            if (keypoints[i, 5, 2] == 0 or keypoints[i, 6, 2] == 0):
                neck = [0, 0, 0]
            limb_index = -1
            for limb in skeleton:
                limb_index += 1
                start_index, end_index = limb  # connection joint index from 0 to 16
                if (start_index == -1):
                    Joint_start = neck
                else:
                    Joint_start = keypoints[i][start_index]
                if (end_index == -1):
                    Joint_end = neck
                else:
                    Joint_end = keypoints[i][end_index]
                # both are Annotated
                # Joint:(x,y,v)
                if ((Joint_start[2] != 0) & (Joint_end[2] != 0)):
                    # print(color)
                    cv2.line(image, tuple(Joint_start[:2]), tuple(Joint_end[:2]), limb_colors[limb_index],3)
        #mask = masks[:, :, i]
        #image = visualize.apply_mask(image, mask, color)
        caption = "{} {:.3f}".format(class_names[class_ids[i]], scores[i])
        cv2.putText(image, caption, (x1 + 5, y1 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color)
    return image

def load(imageDataName):      #一般物体のpickleをロード
    with open(imageDataName, 'rb') as f:
        imageData = pickle.load(f)
    #print("lload")

    image = imageData.image
    roi = imageData.roi
    classID = imageData.classID
    score = imageData.score
    #print("loadMid")

    

    return imageData

def save():

    return

frame = cv2.imread("baske.jpg")
#imageData = load("baske.jpg.pickle")
#frame = imageData.image
#cv2.imshow('frame', frame)

results = model.detect_keypoint([frame], verbose=0)
r = results[0]
# for one image
log("rois", r['rois'])
print("keypoints", r['keypoints'])
log("keypoints", r['keypoints'])
"""
keypoints:全キーポイント
keypoints[0]:人ひとりの全キーポイント
keypoints[0][0]:そのうちの1つのキーポイント
keypoints[0][0][0]:x座標？
keypoints[0][0][1]:y座標？
keypoints[0][0][2]:検出できたかどうか？(1or2)
shape:(4,17,3)
(検出された人の数、キーポイント数(固定)、（座標とboolean）)
"""
print("keypoints[0]",r['keypoints'][0])
print("keypoints[0][0]",r['keypoints'][0][0])
print("keypoints[0][0][0]",r['keypoints'][0][0][0])
log("class_ids", r['class_ids'])
log("keypoints", r['keypoints'])
#log("masks", r['masks'])
log("scores", r['scores'])


result_frame = cv2_display_keypoint(frame,r['rois'],r['keypoints'],
                            r['class_ids'],r['scores'],class_names)

cv2.imshow('frame', result_frame)
cv2.waitKey()

cv2.destroyAllWindows()

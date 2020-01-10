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

#from mrcnn2 import utils
#import mrcnn2.model as modellib
from mrcnn2 import visualize

# Import COCO config
#import coco

import IoU_getter
from IoU_getter import Get_IoU
import vector_getter
from vector_getter import Get_vector
import cv2
import copy



#from ..Mask_RCNN_HumanPose import visualize_demo
#from visualize_demo import Keypoint_result


import copy

class Keypoint_result:
    def __init__(self, image, roi, keypoint, classID, score, file_name):
        self.image = image
        self.keypoint = keypoint
        self.roi = roi
        self.classID = classID
        self.score = score
        self.file_name = file_name
        self.sizeX = len(image[0])
        self.sizeY = len(image)
        return


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

class LoadAndComparison:
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
    
    skeleton = [0,-1,-1,5,-1,6,5,7,6,8,7,9,8,10,11,13,12,14,13,15,14,16]


    def __init__(self):
        self.getIoU = Get_IoU(self.class_names)
        #self.getVector = Get_vector(self.class_names)

        return

  
    """
    def IoU_compute(self, data1, data2, way_of_compute, width = 1000, height = 1000): 
    
    #data1 = self.load(imDataName1, False)                #画像１
    #data2 = self.load(imDataName2, False)                #画像２
    

    ratio1 = self.roiNormalize(imData = data1, 
                            width = width, height = height) #正規化 
    ratio2 = self.roiNormalize(imData = data2, 
                            width = width, height = height) #正規化 

    roi1_orig = data1.roi
    roi2_orig = data2.roi
    data1.roi = ratio1
    data2.roi = ratio2
    """
    #dataの中身を書き換えていいのか？
    #面積自体は後で計算可能なのでたぶん問題ない

    """
    sum, IoUs = self.getIoU.IoU_comparison(data1, data2)
                #比較処理の実行
                #IoUの単純可算sumと、IoUの組み合わせの配列IoUsを得る    
                #roiとclassIDの値は変わらない
                #IoUsの要素番号はvector処理で要素が消されたあとの番号を格納
                #→vector処理後のroiとclassIDを得る必要がある
                #→処理を分割する？

    
    """
    #LastSUM 0.74           #スカラー量
    #LastIoUs [[[1, 1], 0.49], [[0, 0], 0.25]]   
                                #[[class1,class2],iou],[[class1,class2],iou]
    """

    
    classID1 = data1.classID
    roi1 = data1.roi
    sizeX1 = data1.sizeX
    sizeY1 = data1.sizeY
    area1 = sizeX1 * sizeY1
    roiArea1 = []
    roiArea_sum1 = 0
    weight1 = []


    classID2 = data2.classID
    roi2 = data2.roi
    sizeX2 = data2.sizeX
    sizeY2 = data2.sizeY
    area2 = sizeX2 * sizeY2
    roiArea2 = []
    roiArea_sum2 = 0
    
    weight2 = []


    
    #roiの各面積を決定し、roiArea1, roiArea2にいれていく
    for ro1 in roi1:                        #正規化後のroi
        area = (ro1[3] - ro1[1]) * (ro1[2] - ro1[0])
        roiArea_sum1 += area
        roiArea1.append(area)           

    for ro2 in roi2:
        area = (ro2[3] - ro2[1]) * (ro2[2] - ro2[0])
        roiArea_sum2 += area            
        roiArea2.append(area)


    #重みを決定し、weight1, weight2にいれていく
    for roa in roiArea1:
        weigh = roa / roiArea_sum1  #正規化後のroiを検出されたroiの全面積(or演算))で割る
        weight1.append(weigh) 

    for roa in roiArea2:
        weigh = roa / roiArea_sum2
        weight2.append(weigh)

    #ここから各々の計算方法をためす

    #可算or相乗 , 重みづけありorなし,
    #最後にルート(もしくは除算)をとるorとらない
    # 2回(順番入れ替えて)けいさんするorしない（多分する）
    #画像自体の縦横比をどうするか
    #ベクトルの類似度は？



    #方法1：可算平均
    if way_of_compute == 1:
        similarity = sum / len(roi1)
        difference = 0

    elif way_of_compute == 2:

        #方法2：各値にdata1のサイズに応じた重みを付ける
        sim_B = 0
        diff_B = 0
        for iou in IoUs:
            sim_B += iou[1] * weight1[iou[0][0]]
                                    #iou[1]:iouの値
                                    #iou[0][0]:使った矩形の番号(data1)
                                    #iou[0][1]:使った値(data2)
        
        similarity = sim_B
        difference = diff_B
    else:
        print("NO WAY COMPUTE")
        return None



    #print("SIMILARITY", similarity)
    data1.roi = roi1_orig 
    data2.roi = roi2_orig 

    return similarity, difference
    """
    """
    def det_with_comparison_roi(self, data1, data2):
        #中心矩形の決定、サイズ合わせ
        width = 1000
        height = 1000

        center_x, center_y = self.det_center_of_gravity(data1)
            #data1のroiの重心位置

        center_roi_number = self.det_center_roi(data1, center_x, center_y)
            #data1の中心矩形の番号
        
        #data1の中心矩形からの各矩形の距離(or角度)を算出
        vector = Get_vector()


        ratio1 = self.roiNormalize(imData = data1, 
                                width = width, height = height) #正規化 
        ratio2 = self.roiNormalize(imData = data2, 
                                width = width, height = height) #正規化 


        return ratio1, ratio2, width, height
    
    def roi_relative_point(self, data, center_roi_number, vector):
        #中心矩形からの各矩形の相対的な距離or角度を求める
        #中心矩形の中心から、各矩形の左上、右下までのベクトルを取る
        #とったベクトルの類似度で矩形の選択と同様の方法で対応するベクトルを求める
        sim = []
        center_roi = data.roi[center_roi_number]
        center_x = (center_roi[1] + center_roi[3]) / 2
        center_y = (center_roi[0] + center_roi[2]) / 2
        for roi in data.roi:
            upper_vector = np.ndarray([center_y, center_x , roi[0], roi[1]])
            lower_vector = np.ndarray([center_y, center_x , roi[2], roi[3]])
            upper_cos = vector.vector_cos(upper_vector)
            lower_cos = vector.vector_cos(lower_vector)





        return

    def det_center_roi(self, data, center_of_gravity_x, center_of_gravity_y):
        #重心から最も近い重心をもつ矩形を中心矩形とする

        nearest_similarity = 0 
        nearest_roi_number = -1
        max_distance = (data.sizeX ** 2 +  data.sizeY ** 2) ** (1/2) / 2
                    #重心との最大距離(画像の対角線の長さの半分)
        i = 0
        for roi in data.roi:
            roi_center_x = (roi[3] + roi[1]) / 2
            roi_center_y = (roi[2] + roi[0]) / 2

            distance_x = abs(center_of_gravity_x - roi_center_x)
            distance_y = abs(center_of_gravity_y - roi_center_y)
            distance = (distance_x ** 2 + distance_y ** 2) ** (1/2) #矩形と重心の距離
            area = (roi[3] - roi[1]) * (roi[2] - roi[0])

            similarity = (1 - distance / max_distance) * area
            
                    #類似度：面積の重みをつけた距離の類似度
            if nearest_similarity < similarity:
                nearest_similarity = similarity
                nearest_roi_number = i
            i += 1

        return nearest_roi_number
    
    def det_center_of_gravity(self, data):
        #重心を求める
            #矩形の中点に長さに比例した重みを与え、合計して画像の長さで割る
        
        roi_area_sum_x = 0    #長さの合計
        roi_area_sum_y = 0    #の合計
        for roi in data.roi:
            roi_x_len = roi[3] - roi[1]                 #矩形の長さ(x)
            roi_x_center =  (roi[3] + roi[1]) / 2       #矩形の中点(ⅹ)
            roi_area_sum_x += roi_x_center * roi_x_len  #重みつき重心

            
            roi_y_len = roi[2] - roi[0]
            roi_y_center =  (roi[2] + roi[0]) / 2
            roi_area_sum_y += roi_y_center * roi_y_len
        
        center_x = roi_area_sum_x / data.sizeX
        center_y = roi_area_sum_y / data.sizeY

        return center_x, center_y
     
    """


    def similarity_compute(self, data1, data2, way_of_compute): 
        similarity = 0
        difference = 0  #difference定義してない場合がある

      
    
        
        classID1 = data1.classID
        roi1 = data1.roi
        sizeX1 = data1.sizeX
        sizeY1 = data1.sizeY
        area1 = sizeX1 * sizeY1
        roiArea1 = []
        roiArea_sum1 = 0
        weight1 = []
        diagonal1 = (sizeX1 ** 2 + sizeY1 ** 2) ** (1/2)

        classID2 = data2.classID
        roi2 = data2.roi
        sizeX2 = data2.sizeX
        sizeY2 = data2.sizeY
        area2 = sizeX2 * sizeY2
        roiArea2 = []
        roiArea_sum2 = 0
        diagonal2 = (sizeX2 ** 2 + sizeY2 ** 2) ** (1/2)
        
        weight2 = []

        dia =  2 * (data1.sizeX ** 2 + data1.sizeY ** 2) ** (1/2) 


        
        #roiの各面積を決定し、roiArea1, roiArea2にいれていく
        for ro1 in roi1:                        #正規化後のroi
            area = (ro1[3] - ro1[1]) * (ro1[2] - ro1[0])
            roiArea_sum1 += area
            roiArea1.append(area)           

        for ro2 in roi2:
            area = (ro2[3] - ro2[1]) * (ro2[2] - ro2[0])
            roiArea_sum2 += area            
            roiArea2.append(area)


        #重みを決定し、weight1, weight2にいれていく
        for roa in roiArea1:
            weigh = roa / roiArea_sum1  #正規化後のroiを検出されたroiの全面積(or演算))で割る
            weight1.append(weigh) 

        for roa in roiArea2:
            weigh = roa / roiArea_sum2
            weight2.append(weigh)

        #ここから各々の計算方法をためす

        #可算or相乗 , 重みづけありorなし,
        #最後にルート(もしくは除算)をとるorとらない
        # 2回(順番入れ替えて)けいさんするorしない（多分する）
        #画像自体の縦横比をどうするか
        #ベクトルの類似度は？

   

        #方法1：可算平均
        if way_of_compute == 1:
            sum, IoUs = self.getIoU.IoU_comparison(data1, data2)
            similarity = sum / len(roi1)
            difference = 0

        elif way_of_compute == 2:
            sum, IoUs = self.getIoU.IoU_comparison(data1, data2)
            #方法2：各値にdata1のサイズに応じた重みを付ける
            sim_B = 0
            diff_B = 0
            for iou in IoUs:
                sim_B += iou[1] * weight1[iou[0][0]]
                                        #iou[1]:iouの値
                                        #iou[0][0]:使った矩形の番号(data1)
                                        #iou[0][1]:使った値(data2)
            
            similarity = sim_B
            difference = diff_B
        elif way_of_compute == 3:
            #方法1-1:外積の平均
            cp_sum, cross_products = self.getIoU.IoU_comparison(data1, data2,
                                    way_of_compute = "cross_product_size")
            """
            similarity = 0
            for val in cross_products:
                similarity += 1 - val[1] / (width * height)
                #print("value",1 - val[1] / (width * height))

            similarity /= roi2.shape[0]
            difference = 0
            """
            similarity = cp_sum / len(roi1)
        
        
        elif way_of_compute == 4:

            #方法2-2：外積の重み付き平均
            cp_sum, cross_products = self.getIoU.IoU_comparison(data1, data2,
                                    way_of_compute = "cross_product_size")
            """
            similarity = 0
            for val in cross_products:
                similarity += (1 - val[1] / (width * height)) * weight1[val[0][0]]
                #print("weight:",weight1[val[0][0]])
                #print("value",(1 - val[1] / (width * height)))
            
            difference = 0
            """
            similarity = 0
            for val in cross_products:
                similarity += val[1] * weight1[val[0][0]]
                #print("weight:",weight1[val[0][0]])
                #print("value",(1 - val[1] / (width * height)))


        elif way_of_compute == 5:
            #方法3-1：マンハッタン距離の平均
            md_sum, manhattan_dists = self.getIoU.IoU_comparison(data1, data2,
                                    way_of_compute = "manhattan_dist")
            """
            similarity = 0
            for val in manhattan_dists:
                similarity += 1 - val[1] / (width + height)
                #print("value",1 - val[1] / (width * height))

            similarity /= roi2.shape[0]
            difference = 0
            """
            similarity = md_sum / len(roi1)



        elif way_of_compute == 6:
            #方法3-2：マンハッタン距離の重み付き平均
            
            md_sum, manhattan_dists = self.getIoU.IoU_comparison(data1, data2,
                                    way_of_compute = "manhattan_dist")
            """
            similarity = 0
            difference = 0
            for val in manhattan_dists:
                similarity += (1 - val[1] / (width + height)) * weight1[val[0][0]]
                                        #iou[1]:iouの値
                                        #iou[0][0]:使った矩形の番号(data1)
                                        #iou[0][1]:使った値(data2)
            """
            similarity = 0
            difference = 0
            for val in manhattan_dists:
                similarity += val[1] * weight1[val[0][0]]
                                        #iou[1]:iouの値
                                        #iou[0][0]:使った矩形の番号(data1)
                                        #iou[0][1]:使った値(data2)
            
            if 1 < similarity:
                print("over!_sim", similarity)
            elif similarity < 0:
                print("under_sim", similarity)

                
            
        elif way_of_compute == 7:       #ベクトルの長さ
            av_sum, add_vector_len = self.getIoU.IoU_comparison(data1, data2,
                                    way_of_compute = "add_vector_len")
          
            """
            print(add_vector_len)
            print("diagonal",dia)
            print(sizeX1)
            print(sizeY1)
            """
            """
            similarity = 0
            for val in add_vector_len:
                similarity += 1 - val[1] / dia


            similarity /= roi2.shape[0]
            difference = 0
            """
            similarity = av_sum / len(roi1)

        elif way_of_compute == 8:       #ベクトルの長さ
            av_sum, add_vector_len = self.getIoU.IoU_comparison(data1, data2,
                                    way_of_compute = "add_vector_len")
          
            """
            print(add_vector_len)
            print("diagonal",dia)
            print(sizeX1)
            print(sizeY1)
            """
            """
            similarity = 0
            for val in add_vector_len:
                similarity += 1 - val[1] / dia


            similarity /= roi2.shape[0]
            difference = 0
            """
            similarity = 0
            difference = 0
            for val in add_vector_len:
                similarity += val[1] * weight1[val[0][0]]
                                        #iou[1]:iouの値
                                        #iou[0][0]:使った矩形の番号(data1)
                                        #iou[0][1]:使った値(data2)

        else:
            print("NO WAY COMPUTE")
            return None



        #print("SIMILARITY", similarity)
        """
        data1.roi = roi1_orig 
        data2.roi = roi2_orig 
        data1.sizeX = sizeX1_orig
        data1.sizeY = sizeY1_orig
        """
        


        return similarity, difference

    """
    def vector_compute(self,data1, data2, way_of_compute,                  #片側計算
                     width = 1000, height = 1000): 
    #ベクトル計算


    ratio1 = self.roiNormalize(imData = data1, 
                                        width = width, height = height) #正規化 
    ratio2 = self.roiNormalize(imData = data2,
                                            width = width, height = height) #正規化 
    roi1_orig = data1.roi
    roi2_orig = data2.roi
    data1.roi = ratio1  #正規化後のroi
    data2.roi = ratio2
    
    
    classID1 = data1.classID
    roi1 = data1.roi
    sizeX1 = data1.sizeX
    sizeY1 = data1.sizeY
    diagonal1 = (sizeX1 ** 2 + sizeY1 ** 2) ** (1/2)
    area1 = sizeX1 * sizeY1
    roiArea1 = []               #roiごとの面積の配列
    roiArea_sum1 = 0            #全矩形の面積の合計
    weight1 = []                #各roiごとの重みの配列

    
    classID2 = data2.classID
    roi2 = data2.roi
    sizeX2 = data2.sizeX
    sizeY2 = data2.sizeY
    diagonal2 = (sizeX1 ** 2 + sizeY2 ** 2) ** (1/2)
    area2 = sizeX2 * sizeY2
    roiArea2 = []
    roiArea_sum2 = 0
    weight2 = []

    dia = 2 * (width ** 2 + height ** 2) ** (1/2)   #width*heightで正規化したときの対角線の長さ
    #print("dia", dia)

    #roiの各面積を決定し、roiArea1, roiArea2にいれていく
    for ro1 in roi1:                        #正規化後のroi
        area = (ro1[3] - ro1[1]) * (ro1[2] - ro1[0])
        roiArea_sum1 += area
        roiArea1.append(area)           

    for ro2 in roi2:
    area = (ro2[3] - ro2[1]) * (ro2[2] - ro2[0])
    roiArea_sum2 += area            
    roiArea2.append(area)


    #重みを決定し、weight1, weight2にいれていく
    for roa in roiArea1:
    weigh = roa / roiArea_sum1  #正規化後のroiを検出されたroiの全面積(or演算))で割る
    weight1.append(weigh) 

    for roa in roiArea2:
    weigh = roa / roiArea_sum2
    weight2.append(weigh)



    if way_of_compute == 1:
    #方法1-1:外積の平均
    cp_sum, cross_products = self.getIoU.IoU_comparison(data1, data2,
                            way_of_compute = "cross_product_size")

    similarity = 0
    for val in cross_products:
        similarity += 1 - val[1] / (width * height)
        #print("value",1 - val[1] / (width * height))

    similarity /= roi2.shape[0]
    difference = 0

    
    
    elif way_of_compute == 2:

    #方法2-2：外積の重み付き平均
    cp_sum, cross_products = self.getIoU.IoU_comparison(data1, data2,
                            way_of_compute = "cross_product_size")

    similarity = 0
    for val in cross_products:
        similarity += (1 - val[1] / (width * height)) * weight1[val[0][0]]
        #print("weight:",weight1[val[0][0]])
        #print("value",(1 - val[1] / (width * height)))
    
    difference = 0
        

    elif way_of_compute == 3:
    #方法3-1：マンハッタン距離の平均
    md_sum, manhattan_dists = self.getIoU.IoU_comparison(data1, data2,
                            way_of_compute = "manhattan_dist")
    similarity = 0
    for val in manhattan_dists:
        similarity += 1 - val[1] / (width + height)
        #print("value",1 - val[1] / (width * height))

    similarity /= roi2.shape[0]
    difference = 0




    elif way_of_compute == 4:
    #方法3-2：マンハッタン距離の重み付き平均
    
    md_sum, manhattan_dists = self.getIoU.IoU_comparison(data1, data2,
                            way_of_compute = "manhattan_dist")

    similarity = 0
    difference = 0
    for val in manhattan_dists:
        similarity += (1 - val[1] / (width + height)) * weight1[val[0][0]]
                                #iou[1]:iouの値
                                #iou[0][0]:使った矩形の番号(data1)
                                #iou[0][1]:使った値(data2)
        
        
    elif way_of_compute == 5:
    av_sum, add_vector_len = self.getIoU.IoU_comparison(data1, data2,
                            way_of_compute = "add_vector_len")
    
    """
    #print(add_vector_len)
    #print("diagonal",dia)
    #print(sizeX1)
    #print(sizeY1)
    """
    similarity = 0
    for val in add_vector_len:
        similarity += 1 - val[1] / dia


    similarity /= roi2.shape[0]
    difference = 0

    else:
    print("NO_WAY_COMPUTE")
    data1.roi = roi1_orig 
    data2.roi = roi2_orig
    return None

    

    data1.roi = roi1_orig 
    data2.roi = roi2_orig

    return similarity, difference
        
    """


    def similarity_difference(self,data1, data2, should_best_weight_compute,
                             keypoint_weight, is_disclete_sim = False):
        coeff = 1000000
        similarity, keypoint_weight = self.getIoU.get_similarity(data1, data2, 
                            should_best_weight_compute, keypoint_weight, is_disclete_sim)
        
        similarity = int(similarity * coeff)
        similarity /= coeff


        return similarity, keypoint_weight






    def roiNormalize(self, imData, width, height):       #roiの値を画像サイズで正規化
        """
        aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
        width,heightの初期値はハードコーディングするしかない
        (そうでないといちいちwidth = って書く必要がある)
        """

        ratio = []                 #roi(矩形)の配列を画像サイズを正規化したときの値に直す(2次元配列) 
                                    #[[x1,y1,x2,y2], [x1,y1,x2,y2], [x1,y1,x2,y2]]
        for ro in imData.roi:
            eachRatio = []

            eachRatio.append(int(height*ro[0]/imData.sizeY)) 
            eachRatio.append(int(width*ro[1]/imData.sizeX))
            eachRatio.append(int(height*ro[2]/imData.sizeY))
            eachRatio.append(int(width*ro[3]/imData.sizeX))
        
            ratio.append(eachRatio)
        ratio = np.array(ratio)

        return ratio

    def load(self, imageDataName, isDisplay = True):
        #path = os.getcwd()
        #print("sssssssssssssssssssssssssssssssss",path)
        os.chdir('./pickles')
        #path = os.getcwd()
        #print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",path)
        #print("load")
        with open(imageDataName, 'rb') as f:
            imageData = pickle.load(f)
        #print("lload")

        image = imageData.image
        roi = imageData.roi
        classID = imageData.classID
        score = imageData.score
        #print("loadMid")

        if isDisplay:
            visualize.display_instances(image, roi, classID, 
                                    self.class_names, scores = score, ax=None)
        #print("end")
        os.chdir('../')


        return imageData


    def load_keypoint(self, keypoint_pickle_name, is_display = True):
        #path = os.getcwd()
        #print("sssssssssssssssssssssssssssssssss",path)
        #os.chdir('./pickle_keypoint')
        path = os.getcwd()
        #print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",path)
        #print("load")
        with open("./pickle_keypoint/{}".format(keypoint_pickle_name), 'rb') as f:

            keypoint_result = pickle.load(f)
        #print("lload")

        keypoint_result.keypoint = keypoint_result.keypoint.astype(np.int)


        #print("loadMid")

        if is_display:
            image = keypoint_result.image
            roi = keypoint_result.roi
            keypoint = keypoint_result.keypoint
            classID = keypoint_result.classID
            score = keypoint_result.score
            #plt.imshow(image)
            #plt.show()
            self.cv2_display_keypoint(image,roi,keypoint,
                                classID,score,self.class_names)
         
            
        #print("end")
        #os.chdir('../')


        return keypoint_result
    def load_kouzu_data(self, keypoint_pickle_name, is_display = True):
        #path = os.getcwd()
        #print("sssssssssssssssssssssssssssssssss",path)
        #os.chdir('./pickle_keypoint')
        path = os.getcwd()
        #print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",path)
        #print("load")
        with open("./kouzu_data/{}".format(keypoint_pickle_name), 'rb') as f:
            keypoint_result = pickle.load(f)
        #print("lload")

        keypoint_result.keypoint = keypoint_result.keypoint.astype(np.int)


        #print("loadMid")

        if is_display:
            image = keypoint_result.image
            roi = keypoint_result.roi
            keypoint = keypoint_result.keypoint
            classID = keypoint_result.classID
            score = keypoint_result.score
            #plt.imshow(image)
            #plt.show()
            self.cv2_display_keypoint(image,roi,keypoint,
                                classID,score,self.class_names)
         
            
        #print("end")
        #os.chdir('../')


        return keypoint_result

    def cv2_display_keypoint(self, image,boxes,keypoints,class_ids,scores,class_names):
        skeleton = self.skeleton
        copy_image = copy.copy(image)
        # Number of persons
        N = boxes.shape[0]
        if not N:
            print("\n*** No persons to display *** \n")
        else:
            assert N == keypoints.shape[0] and N == class_ids.shape[0] and N==scores.shape[0],\
                "shape must match: boxes,keypoints,class_ids, scores"
        colors = visualize.random_colors(N)
        for i in range(N):
            color = colors[i]
            # Bounding box
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            cv2.rectangle(copy_image, (x1, y1), (x2, y2), color, thickness=2)
            for Joint in keypoints[i]:
                if (Joint[2] != 0):
                    cv2.circle(copy_image,(Joint[0], Joint[1]), 2, color, -1)

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
                        cv2.line(copy_image, tuple(Joint_start[:2]), tuple(Joint_end[:2]), limb_colors[limb_index],3)
            #mask = masks[:, :, i]
            #image = visualize.apply_mask(image, mask, color)
            #caption = "{} {:.3f}".format(class_names[class_ids[i]], scores[i])
            caption = "{}".format(class_names[class_ids[i]])
            cv2.putText(copy_image, caption, (x1 , y1), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color)
            
        plt.imshow(copy_image)
        plt.show()
        return 

    
   
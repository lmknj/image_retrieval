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
import glob


#from mrcnn2 import utils
#import mrcnn2.model as modellib
from mrcnn2 import visualize

# Import COCO config
#import coco

import loadAndComparison as Lac
from loadAndComparison import Keypoint_result
from loadAndComparison import LoadAndComparison
from loadAndComparison import ImageData
#from ..Mask_RCNN_HumanPose import visualize_demo
#from visualize_demo import Keypoint_result

import IoU_getter
from IoU_getter import Get_IoU
import vector_getter
from vector_getter import Get_vector

import math
import cv2
import copy



class Visualize_IoU:
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

    def __init__(self, keypoint_pickle_name, is_kouzu_data):
        self.lac = LoadAndComparison()

        if is_kouzu_data:    
            self.keypoint_result = self.lac.load_kouzu_data(
                            keypoint_pickle_name = keypoint_pickle_name,
                            is_display= False)
        else:
            self.keypoint_result = self.lac.load_keypoint(
                        keypoint_pickle_name = keypoint_pickle_name,
                        is_display= False)
        

        #print(self.keypoint_result.classID)

        self.is_kouzu_data = is_kouzu_data




        return
    def only_keypoint(self):
        a = self.keypoint_result
        i = 0
        for classID, keypoint in zip(a.classID, a.keypoint):
            print("number", i)
            print("classID", classID)
            print("keypoint", keypoint)
            i += 1

        return

    def while_token(self, size_range, roi_range, keypoint_range, roi_move_range, times, 
                         append_times, remove_times, way_of_compute):

        #keypoint_resultは__init__で開いたデータのどちらか(1or2)
        #times:token画像生成枚数
        #append_times = 画像内に増やす矩形
        """
        keypoint_resultを微妙に変えた画像を生成する
        矩形の数は変えない
         size_range ：画像のサイズ比(x,y)変更の係数(10なら元の画像のサイズの1/10のサイズの範囲内で変化する)
        roi_ramge : ↾の矩形バージョン
        append_times:矩形を追加する数
        remove_times：矩形を削除する数
        """

        keypoint_result = self.keypoint_result

            
        sizeX = keypoint_result.sizeX
        sizeY = keypoint_result.sizeY
        #image = keypoint_result.image
        image = self.generateWhite(sizeX, sizeY,isBlack = False)    #真っ黒な画像内に矩形を表示する
        roi = keypoint_result.roi
        classID = keypoint_result.classID
        score = keypoint_result.score
        keypoint = keypoint_result.keypoint

        keypoint_result = self.modify_keypoint(keypoint_result)


        

        N = roi.shape[0] + append_times    #２つの画像の矩形の色を統一する
        colors = visualize.random_colors(N)
        #print("色",colors)

     
        #visualize.display_instances(keypoint_result.image, roi, classID, self.class_names,
        #                             scores = score, ax=None, colors = colors,
        #                             save = True, file_name = "tokenNotImage",) #変更前
        self.lac.cv2_display_keypoint(keypoint_result.image,roi,keypoint,
                                classID,score,self.class_names)
        
        sim_and_tokenData = []

        for i in range(times):
            #print(i)
            imToken = copy.deepcopy(keypoint_result)
            keypoint_resultToken, similarity, difference = self.generate_token_and_comparison(
                        keypoint_result = imToken, colors = colors, size_range = size_range, 
                        roi_range = roi_range, keypoint_range = keypoint_range,
                        roi_move_range = roi_move_range, append_times = append_times, 
                        remove_times = remove_times, name_count = i, way_of_compute = way_of_compute)
            
            each = [keypoint_resultToken, similarity, difference]

            sim_and_tokenData.append(each)

        
        
        return sim_and_tokenData

    

    def generate_token_and_comparison(self, keypoint_result, colors, size_range, roi_range,
            keypoint_range, roi_move_range, append_times, remove_times, name_count, way_of_compute):
        
        keypoint_resultToken = self.generate_token(keypoint_result, colors, size_range, roi_range,
                    keypoint_range, roi_move_range, append_times, remove_times, name_count, way_of_compute)
        
        image = keypoint_resultToken.image
        roi = keypoint_resultToken.roi
        keypoint = keypoint_resultToken.keypoint
        classID = keypoint_resultToken.classID
        score = keypoint_resultToken.score
        file_name = keypoint_resultToken.file_name
        
        """
        print("image",type(image))
        print("roi",type(roi))
        print("keypoint",type(keypoint))
        print("classID",type(classID))
        print("score",type(score))
        """

        similarity, difference = self.lac.similarity_difference(keypoint_resultToken, keypoint_result, way_of_compute)
                        
        
        ###########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!順番変更不可#######################
        print("total_similarity",similarity)
     
        self.lac.cv2_display_keypoint(image,roi,keypoint,
                               classID,score,self.class_names)

        #cv2_diaplay_keypointの引数：(self, image,boxes,keypoints,class_ids,scores,class_names)
        #visualize.display_instances(image, roi, classID, self.class_names,
        #                scores = score, ax=None, colors = colors, file_name = file_name,
        #                save = True, similarity = similarity, difference = difference)


        #print(file_name)

        
        return keypoint_resultToken, similarity, difference

    def generate_token_parallel_displace(self, keypoint_result, roi_move_range):  #微妙に変えた画像
        #generate_white関数のアルゴリズム変更版
        #物体の位置だけを統一的に移動する


        #ここから、サイズを変更し，そのサイズに合わせてroiのサイズを合わせる


        roi_rand = []    #2次元配列(roiと対応))
                                #[0]:矩形と対応
                                #[0][0]:1つ目の矩形の左上のｙ座標

       
        if roi_move_range == 0:
            roi_x_range = 0
            roi_y_range = 0
        else:
            roi_x_range = random.randrange(-roi_move_range , roi_move_range)
            roi_y_range = random.randrange(-roi_move_range , roi_move_range)
            
        for roi, pose in zip(keypoint_result.roi, keypoint_result.keypoint):      #ro[y0, x0, x0, x1]
            
            roi[0] += roi_y_range #平行移動
            roi[1] += roi_x_range
            roi[2] += roi_y_range
            roi[3] += roi_x_range

            if roi[3] > keypoint_result.sizeX: #右下のｘが画像サイズを超えてたら
                roi[3] = keypoint_result.sizeX

            if roi[2] > keypoint_result.sizeY: #右下のyが画像サイズを超えてたら
                roi[2] = keypoint_result.sizeY

            if roi[1] > roi[3]: #左上のｘが右下のxを超えてたら
                roi[1] = roi[3]-1

            if roi[0] > roi[2]: #左上のyが右下のyを超えてたら
                roi[0] = roi[2]-1
            


            #キーポイントのランダム変更
            for each_point in pose: 
                                    #pose[[x,y bool], [x, y, bool],...]
                                    #each[x,y,,bool]
                if each_point[-1]:  #表視されているなら
                    #roiに合わせる
                    each_point[0] += roi_x_range
                    each_point[1] += roi_y_range

                    
                    if each_point[0] > roi[3]: #pointがyが矩形の右端を超えてたら
                        each_point[0] = roi[3]-1

                    if each_point[1] > roi[2]: #pointがyが矩形の下端を超えてたら
                        each_point[1] = roi[2] - 1

                    if each_point[0] < roi[1]: #pointがyが矩形の左端を超えてたら
                        each_point[0] = roi[1] + 1

                    if each_point[1] < roi[0]: #pointがyが矩形の上端を超えてたら
                        each_point[1] = roi[0] + 1
                    

        
        return keypoint_result


    def modify_keypoint(self, keypoint_result):  #微妙に変えた画像
        #generate_white関数のアルゴリズム変更版
        #物体の位置だけを統一的に移動する


        #ここから、サイズを変更し，そのサイズに合わせてroiのサイズを合わせる

            
        for roi, pose in zip(keypoint_result.roi, keypoint_result.keypoint):      #ro[y0, x0, x0, x1]
            

            #キーポイントのランダム変更
            for each_point in pose: 

                if each_point[0] > roi[3]: #pointがyが矩形の右端を超えてたら
                    each_point[2] = 0

                if each_point[1] > roi[2]: #pointがyが矩形の下端を超えてたら
                    each_point[2] = 0

                if each_point[0] < roi[1]: #pointがyが矩形の左端を超えてたら
                    each_point[2] = 0

                if each_point[1] < roi[0]: #pointがyが矩形の上端を超えてたら
                    each_point[2] = 0
                

        
        return keypoint_result


    def generate_token(self, keypoint_result, colors, size_range, roi_range, keypoint_range,
                    roi_move_range, append_times, remove_times, name_count, way_of_compute):  #微妙に変えた画像
        """
        keypoint_resultを微妙に変えた画像を生成する
        矩形の数は変えない
        size_range ：画像のサイズ比(x,y)変更の係数(10なら元の画像のサイズの1/10のサイズの範囲内で変化する)
        roi_ramge : ↾の矩形バージョン
        keypoint_range :キーポイントの角度の変更度合い(0~1)
        append_times:矩形を追加する数
        remove_times：矩形を削除する数
        name_count：トークンデータの後ろにつける番号
        
        """
        
        sizeX = keypoint_result.sizeX
        sizeY = keypoint_result.sizeY
        #image = keypoint_result.image
        image = self.generateWhite(sizeX, sizeY,isBlack = False)    #真っ黒な画像内に矩形を表示する
        roi = keypoint_result.roi
        classID = keypoint_result.classID
        score = keypoint_result.score
        keypoint = keypoint_result.keypoint


        
     


        #ここから、サイズを変更し，そのサイズに合わせてroiのサイズを合わせる

        if int(sizeX/size_range) <= 1:
            x_rand = sizeX
        else:
            x_rand = sizeX + random.randrange(-int(sizeX/size_range), int(sizeX/size_range))     #画像サイズ変更
        
        if int(sizeY/size_range) <= 1:
            y_rand = sizeY
        else:
            y_rand = sizeY + random.randrange(-int(sizeY/size_range), int(sizeY/size_range))
        image = self.generateWhite(x_rand, y_rand,isBlack = False)

        roi = roi.tolist()              #listに変換
        classID = classID.tolist()      #〃
        score = score.tolist()
        keypoint = keypoint.tolist()

        while append_times > 0:                                     #矩形をランダムに追加する処理
        
            token_classID = random.randrange(0, len(self.class_names))  #クラス名ランダム
            classID.append(token_classID)
            
            x_len = random.randrange(50, 150)  #適当な範囲
            y_len = random.randrange(50, 150)  #適当な範囲

            lowerY = random.randrange(y_len, y_rand)
            lowerX = random.randrange(x_len, x_rand)
            
            upperY = lowerY - y_len
            upperX = lowerX - x_len

            token_roi = np.array([upperY, upperX, lowerY, lowerX])

            token_keypoint = np.array([[
            [500,100,1],
            [400,50,1],
            [600,50,1],
            [350,100,1],
            [650,100,1],
            [100,200,1],
            [900,200,1],
            [100,400,1],
            [900,400,1],
            [100,600,1],
            [900,600,1],
            [400,600,1],
            [600,600,1],
            [400,800,1],
            [600,800,1],
            [400,1000,1],
            [600,1000,1]
        
            ]], dtype = np.int32)       #appendするkeypoint仮

            roi.append(token_roi)

            score.append(-1.0)

            keypoint.append(token_keypoint)



            append_times -= 1

        roi = np.array(roi)
        classID = np.array(classID)
        score = np.array(score)
        keypoint = np.array(keypoint)

       


        roi_rand = []    #2次元配列(roiと対応))
                                #[0]:矩形と対応
                                #[0][0]:1つ目の矩形の左上のｙ座標

       


            
        for ro, pose in zip(roi, keypoint):      #ro[y0, x0, x0, x1]
            roi_val = []
            count = 0
                  #(画像位置の1/rangeだけサイズ変更)
            x_len = ro[3] - ro[1]   # 矩形のｘの長さ
            y_len = ro[2] - ro[0]   #yの〃

            #roi_center = np.ndarray([(ro[3] + ro[1]) / 2, (ro[2] + ro[0]) / 2]) 
            
            roi_center = [(ro[3] + ro[1]) / 2, (ro[2] + ro[0]) / 2]
                         
                                #矩形の中心位置(x,y)

            
            for val in ro:
                if roi_range > 0:

                    if count % 2 == 0:   #左上のYか右下のYなら
                        value = val + random.randrange(-1 * math.ceil(y_len/roi_range), 
                                                            math.ceil(y_len/roi_range)) 
                                                            #ceil:最小の整数
                                                            #ランダムで変更
                        value = int(y_rand*value / sizeY)   #変更後の画像サイズで正規化
                    
                    else:
                        #print(int(x_len/roi_range))
                        #print("x_len",x_len)
                        #print("roi",roi_range)

                        value = val + random.randrange(-1 * math.ceil(x_len/roi_range), 
                                                        math.ceil(x_len/roi_range))

                                                            #ランダムで変更
                        value = int(x_rand*value / sizeX)   
                else:
                    value = val


                roi_val.append(value)     #結果を追加
                

                count += 1
            
            if roi_val[3] > x_rand: #右下のｘが画像サイズを超えてたら
                roi_val[3] = x_rand

            if roi_val[2] > y_rand: #右下のyが画像サイズを超えてたら
                roi_val[2] = y_rand

            if roi_val[1] > roi_val[3]: #左上のｘが右下のxを超えてたら
                roi_val[1] = roi_val[3]-1

            if roi_val[0] > roi_val[2]: #左上のyが右下のyを超えてたら
                roi_val[0] = roi_val[2]-1

            roi_val = np.array(roi_val)
            roi_rand.append(roi_val)
            #roi完了
                 
            len_ratio_x = (roi_val[3] - roi_val[1]) / x_len            #長さの比
            len_ratio_y = (roi_val[2] - roi_val[0]) / y_len           #長さの比
            roi_center = [(roi_val[3] + roi_val[1]) / 2, (roi_val[2] + roi_val[0]) / 2]


            #キーポイントのランダム変更
            for each_point in pose: 
                                    #pose[[x,y bool], [x, y, bool],...]
                                    #each[x,y,,bool]
                if each_point[-1]:  #表視されているなら
                    #roiに合わせる
                    eachX_diff = each_point[0] - ro[1] #もとの矩形の左端とキーポイントの差(x)
                    each_point[0] = roi_val[1] +  eachX_diff * len_ratio_x 
                                                #比に基づいてバランスを変更

                    eachY_diff = each_point[1] - ro[0] #左端との差
                    each_point[1] = roi_val[0] +  eachY_diff * len_ratio_y 
                                                #比に基づいてバランスを変更



                    key_rand_value = random.uniform(-1 * keypoint_range * math.pi,
                                            keypoint_range * math.pi) #回転のレンジ
                    each_point_originX = each_point[0] - roi_center[0]
                    each_point_originY = each_point[1] - roi_center[1]      #原点に合わせる
                    each_point_rotationX = each_point_originX * math.cos(key_rand_value) \
                                        - each_point_originY * math.sin(key_rand_value)
                    each_point_rotationY = each_point_originX * math.sin(key_rand_value) \
                                        + each_point_originY * math.cos(key_rand_value)   #回転
                    each_point[0] = each_point_rotationX + roi_center[0]
                    each_point[1] = each_point_rotationY + roi_center[1]    #代入

                    if each_point[0] > roi_val[3]: #pointがyが矩形の右端を超えてたら
                        each_point[0] = roi_val[3]-1

                    if each_point[1] > roi_val[2]: #pointがyが矩形の下端を超えてたら
                        each_point[1] = roi_val[2] - 1

                    if each_point[0] < roi_val[1]: #pointがyが矩形の左端を超えてたら
                        each_point[0] = roi_val[1] + 1

                    if each_point[1] < roi_val[0]: #pointがyが矩形の上端を超えてたら
                        each_point[1] = roi_val[0] + 1

        
        #for文終了でroiの値変更が完了
        #print("pre_ImageSize X:Y", sizeX, sizeY)
        #print("chnged_ImageSize X:Y", x_rand, y_rand)

        #print("pre_roi", roi)

        roi = np.array(roi_rand)

        
    

        #print("changed_roi", roi)
        #print("ROItype:{}".format(type(roi)))
        file_name = "tokenData" + str(name_count)
        keypoint_resultToken = Keypoint_result(image, roi, keypoint, classID, score, file_name)
        #print("imTOKENROItype:{}".format(type(keypoint_resultToken.roi)))
        # def __init__(self, image, roi, keypoint, classID, score, file_name):

        keypoint_resultToken = self. generate_token_parallel_displace(keypoint_resultToken, roi_move_range)
        
        return keypoint_resultToken


   


    def generateWhite(self, width = 1000, height = 1000, isShow = False, isBlack = False):
        #画像のサイズを決めて単色で塗りつぶす
        #np配列を用意すればおｋ？
        #visualizeは
        if not isBlack:
            image = np.full((height, width, 3), 255) #width*heightの白画像生成
        else:
            image = np.zeros((height, width, 3))
        if isShow:
            plt.imshow(image)
            plt.show()

        return image

    def visualize_image(self):
        keypoint_result = self.keypoint_result

        if self.is_kouzu_data:
            image = np.where(keypoint_result.image <= 255, 255, 255)    #白に変える
        else:
            image = keypoint_result.image
        self.lac.cv2_display_keypoint(image,
                                    keypoint_result.roi,
                                    keypoint_result.keypoint,
                                    keypoint_result.classID,
                                    keypoint_result.score,
                                    self.class_names)



        return


   

def main():
    #os.chdir('./images')
    #vi = Visualize_IoU("kouzudata_yoshii_16.pickle" , is_kouzu_data = True)

    vi = Visualize_IoU("037438278.jpg__keypoint__.pickle" , is_kouzu_data = False)#Befit
    #vi = Visualize_IoU("043677717.jpg__keypoint__.pickle" , is_kouzu_data = False)#niku
    

    vi.visualize_image()
    #vi.while_token(times = 1, size_range = 10000, roi_range = 0, keypoint_range = 0.0, 
    #                    roi_move_range = 0, append_times = 0, remove_times = 0, way_of_compute = "vector5")
    #vi.only_keypoint()

    return

if __name__ == "__main__" :
    main()
    
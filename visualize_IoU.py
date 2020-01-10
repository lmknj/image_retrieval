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
from loadAndComparison import ImageData
from loadAndComparison import LoadAndComparison

import IoU_getter
from IoU_getter import Get_IoU
import vector_getter
from vector_getter import Get_vector

import math

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

    def __init__(self, imageDataName1):
        self.lac = LoadAndComparison()



        self.imageData1 = self.lac.load(
                        imageDataName = imageDataName1, isDisplay= False)
       

      
        #print("called")


        return

    def while_token(self, size_range, roi_range, times, 
                         append_times, remove_times, way_of_compute):

        #imageDataは__init__で開いたデータのどちらか(1or2)
        #times:token画像生成枚数
        #append_times = 画像内に増やす矩形
        """
        imagedataを微妙に変えた画像を生成する
        矩形の数は変えない
         size_range ：画像のサイズ比(x,y)変更の係数(10なら元の画像のサイズの1/10のサイズの範囲内で変化する)
        roi_ramge : ↾の矩形バージョン
        append_times:矩形を追加する数
        remove_times：矩形を削除する数
        """

        imageData = self.imageData1

            
        sizeX = imageData.sizeX
        sizeY = imageData.sizeY
        #image = imageData.image
        image = self.generateWhite(sizeX, sizeY,isBlack = False)    #真っ黒な画像内に矩形を表示する
        roi = imageData.roi
        classID = imageData.classID
        score = imageData.score
        

        N = roi.shape[0] + append_times    #２つの画像の矩形の色を統一する
        colors = visualize.random_colors(N)
        #print("色",colors)

     
        visualize.display_instances(imageData.image, roi, classID, self.class_names,
                                     scores = score, ax=None, colors = colors,
                                     save = True, file_name = "tokenNotImage",) #変更前
        
        sim_and_tokenData = []

        for i in range(times):
            #print(i)
            imToken = copy.deepcopy(imageData)
            imageDataToken, similarity, difference = self.generate_token_and_comparison(
                        imageData = imToken, colors = colors, size_range = size_range, 
                        roi_range = roi_range, append_times = append_times, 
                        remove_times = remove_times, name_count = i, way_of_compute = way_of_compute)
            
            each = [imageDataToken, similarity, difference]

            sim_and_tokenData.append(each)

        
        
        return sim_and_tokenData

    def generate_token_and_comparison(self, imageData, colors, size_range, roi_range,
                         append_times, remove_times, name_count, way_of_compute):
        
        imageDataToken = self.generate_token(imageData, colors, size_range, roi_range,
                         append_times, remove_times, name_count, way_of_compute)
        
        image = imageDataToken.image
        roi = imageDataToken.roi
        classID = imageDataToken.classID
        score = imageDataToken.score
        file_name = imageDataToken.file_name


        similarity, difference = self.lac.similarity_difference(
                        imageDataToken, imageData, way_of_compute)
        
        ###########!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!順番変更不可#######################

     

        visualize.display_instances(image, roi, classID, self.class_names,
                        scores = score, ax=None, colors = colors, file_name = file_name,
                        save = True, similarity = similarity, difference = difference)


        #print(file_name)

        
        return imageDataToken, similarity, difference
    
    def generate_token(self, imageData, colors, size_range, roi_range,
                         append_times, remove_times, name_count, way_of_compute):  #微妙に変えた画像
        """
        imagedataを微妙に変えた画像を生成する
        矩形の数は変えない
        size_range ：画像のサイズ比(x,y)変更の係数(10なら元の画像のサイズの1/10のサイズの範囲内で変化する)
        roi_ramge : ↾の矩形バージョン
        append_times:矩形を追加する数
        remove_times：矩形を削除する数
        name_count：トークンデータの後ろにつける番号
        
        """
        
        sizeX = imageData.sizeX
        sizeY = imageData.sizeY
        #image = imageData.image
        image = self.generateWhite(sizeX, sizeY,isBlack = False)    #真っ黒な画像内に矩形を表示する
        roi = imageData.roi
        classID = imageData.classID
        score = imageData.score
     


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

            roi.append(token_roi)

            score.append(-1.0)



            append_times -= 1

        roi = np.array(roi)
        classID = np.array(classID)
        score = np.array(score)


        roi_rand = []    #2次元配列(roiと対応))
                                #[0]:矩形と対応
                                #[0][0]:1つ目の矩形の左上のｙ座標
        
        for ro in roi:      #ro[y0, x0, x0, x1]
            roi_val = []
            count = 0
                  #(画像位置の1/rangeだけサイズ変更)
            x_len = ro[3] - ro[1]   # 矩形のｘの長さ
            y_len = ro[2] - ro[0]   #yの〃

            for val in ro:
                if count % 2 == 0:   #左上のYか右下のYなら
                    value = val + random.randrange(-1 * math.ceil(y_len/roi_range), 
                                                        math.ceil(y_len/roi_range)) 
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
        
        #for文終了でroiの値変更が完了
        #print("pre_ImageSize X:Y", sizeX, sizeY)
        #print("chnged_ImageSize X:Y", x_rand, y_rand)

        #print("pre_roi", roi)

        roi = np.array(roi_rand)

        
    

        #print("changed_roi", roi)
        #print("ROItype:{}".format(type(roi)))
        file_name = "tokenData" + str(name_count)
        imageDataToken = ImageData(image, roi, classID, score, file_name)
        #print("imTOKENROItype:{}".format(type(imageDataToken.roi)))

        return imageDataToken


   


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



    """
    def getRoi(self):
        #imageDataから正規化したroiを切り出したい

        roi1, roi2 = self.lac.roiNormalize(imData1 = self.imageData1,
                    imData2 = self.imageData2,width = 1000, height = 1000)
        
        #print(roi1)
        #print(roi2)

        return roi1, roi2
    def getClassID(self):
        classID1 = self.imageData1.classID
        classID2 = self.imageData2.classID

        return classID1, classID2

    def selectCompRoi(self):
        #適当に決める(後から修正の流れ)
        roi1, roi2 = self.getRoi()
        classID1, classID2 = self.getClassID()
        print("class1",classID1)
        print("class2",classID2)

        selectedRoi1 = roi1[0]              #適当に決めた矩形その１
        selectedRoi2 = roi2[0]              #適当に決めた矩形その2
        print("roi1",roi1)
        print("roi2",roi2)
        print("sroi1",selectedRoi1)
        print("sroi2",selectedRoi2)

        ###################クラス名その他一切考慮していません##############

        #print("sel1",selectedRoi1)
        #print("sel2",selectedRoi2)

        return  selectedRoi1, selectedRoi2

    def detAnd(self):
        roi1, roi2 = self.selectCompRoi()

        andArea = self.lac.detAnd(ratio1 = roi1, ratio2 = roi2)

        andValue = (andArea[2] - andArea[0]) * (andArea[3] - andArea[1])#面積計算

        return andArea

    def boxLining(self, image, roi, color):

        #roi y:[0],[2], x:[1],[3]
        #image[y[x[youso]]]
        
        print(roi)
        
        for y in range(roi[0], roi[2]):#縦方向の塗りつぶし
            image[y][roi[1]]=color      #なんか描画されないときがある
            image[y][roi[3]]=color      #〃
        
        #横方向
        #image[roi[0]+5 ][roi[1]:roi[3]] = color

        image[roi[0]][roi[1]:roi[3]] = color  #なんか描画されないときがある  
        image[roi[2]][roi[1]:roi[3]] = color  #〃




        return
    def boxFill(self,image, roi, color):
        for y in range(roi[0], roi[2]):#縦方向の塗りつぶし
            image[y][roi[1]:roi[3]] = color
            



        return

    def visualize_roiAnd(self):
        image = self.generateWhite()

        roi1, roi2 = self.selectCompRoi()

        
        andArea = self.detAnd()

        green = np.array([0, 255, 0])
        self.boxFill(image, andArea, green)

        
        red = np.array([255, 0, 0])
        

        blue = np.array([0, 0, 255])
        self.boxLining(image, roi1, red)
        self.boxLining(image, roi2, blue)

        
        plt.imshow(image)
        plt.show()


        return
    """

   
"""
def main():
    #os.chdir('./images')
    vi = Visualize_IoU("PPS_mituwosuukuroageha_TP_V.jpg.pickle")
    #vi.visualize_roiAnd()
    vi.while_token(times = 1, size_range = 1000, roi_range = 1000,
                         append_times = 0, remove_times = 0, way_of_compute = "vector5")



    return

main()
"""
    
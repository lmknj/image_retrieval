import os
import sys
import time
import random
import math
import copy

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


class Get_vector:
    def __init__(self):
        pass


    """
    def vector_comparison(self,data1, data2, way_of_compute = "cross_product_size"):
        #画像当たりの相違度が小さい順にソートする
        
        classID1 = data1.classID
        classID2 = data2.classID
        roi1 = data1.roi
        roi2 = data2.roi
        is_sim = True   #相違度か類似度か

        
        


        classTypeList = []  #クラスごとに類似度の最大値か相違度の最小値の和を求めるための配列 
                            #[[sortedVectorList], [sortedVectorList], [sortedVectorList],...]

        #↓2重ループで各矩形の類似度or相違度を総当たりで計算する
        #num_and_vectorに計算に使用した矩形の番号の組み合わせ[i,j]とその類似度or相違度を入れる
        #num_and_vectorの集合([i]が等しいもの同士の集合)をvector_listに入れる
        #それを類似度or相違度に基づいてソートする
        #その集合をclassTypeListに格納する

        for i in range(0, len(classID1)):   #classIDだけ繰り返す
            vector_list = []    #iごとに作成される(i=0に対するj=0,1,2という感じ)[[i,j],類似度or相違度]の配列
                                #[[num_and_vector],[num_and_vector],[num_and_vector],...]


            for j in range(0, len(classID2)):
                num_and_simdiff = []        #data2.classIDの要素番号(j)と
                                            #そのクラスに対応したIoUの配列[クラス番号, 類似度or相違度]の配列
                                            #[クラス番号, 類似度or相違度]

                vectors = self.vector_roi2roi(roi1[i], roi2[j])    #[左上、右下、中心]
                                                                    #data1のi番目の矩形とdata2のj番目の矩形のベクトル 
                                                                   
                                           
                num_and_simdiff.append([i,j])
                
                if classID1[i] == classID2[j]:    #クラス名が等しければ
                    if way_of_compute == "cross_product_size":
                        is_sim = False  #相違度の計算
                        num_and_simdiff.append(self.cross_product_size(data1, vectors[0], vectors[1])) 
                            #左上ベクトルと右下ベクトルの外積を相違度に入れる
                        

                    elif way_of_compute == "cos_by_len":
                        #is_sim = True
                        num_and_simdiff.append(self.cos_by_len(data1, vectors[0], vectors[1])) 

                    elif way_of_compute == "manhattan_dist":
                        is_sim = False
                        num_and_simdiff.append(self.manhattan_dist(data1, vectors[0], vectors[1]))

                    elif way_of_compute == "add_vector_len":
                        num_and_simdiff.append(self.add_vector_len(data1, vectors[0], vectors[1]))

                    else:
                        print("error: invalid way_of_compute")
                        print("finish")
                        return 
                                          
                else:
                    num_and_simdiff.append(-1)                     #クラス名が異なれば
                                                                    #類似度or相違度の代わりに-1を代入
                                                                    #類似度も相違度も不にはならない

                vector_list.append(num_and_simdiff)                   #完成したnum_and_simdiffをvector_listに格納
                #print("NAI",numAndIoU)               
                #print("ssS",i,j)

            #print("ddddddddddddddddddddddddddddddddd")

            #print("sort前vector_list", vector_list)

            sorted_vector_list = self.vector_sort(vector_list, is_sim)  #類似度or相違度に基づいてソート
            
            #print("sort後vector_list", sorted_vector_list)
            #print("IL",IList)
            #print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh ")
                               
            classTypeList.append(sorted_vector_list)     #ソートされたリストをclassTypeListに格納

        #ここまでfor文

        #print("終了後classTypeList", classTypeList)





        #print("CTL",classTypeList)   
            #4次元配列
            #[0]:IoUList：jの各要素と比較numとIoU
            #[0][0]:numAndIoU：文字通り
            #[0][0][0]:iとj
            #[0][0][0][0]:i
            #[0][0][0][1]:j
            #[0][0][1]:↾のIoU
        

        #ソート済み配列sortedIoUListを含んだClassTypeList
        #これを各クラスごとにIOUの合計値が最大になるように計算
            #最も大きいIOU値をを持つiを採用し、
            #そのiが用いたjを排除してさらに再計算する
            #ただしIoUの値が０でその走査は終了→IoU０で算出する

        #これを各クラスごとに適用して、IoUの和を求める
        # 検出された矩形の数で割って類似度を求める
    
        classSize = len(self.class_names) #認識できるクラス数
        
        sum = 0                            #IoUが最大の時の総和
        simdiffs = []

        
        #↓2重ループ
        #各クラスごとに、そのクラスに属する、類似度が最も高いか、相違度が最も小さい値をもつ矩形の組み合わせを決定する
        for i in range(0,classSize):                #全クラスを走査する
            eachClass = []                          #各クラス
            for c in range(0,len(classID1)):        #data1のクラスがどのクラスに属するかを決定する
                if classID1[c] == i:
                    #print(i)
                    #print(c)
                    #print("classTypeList[c]",classTypeList[c])
                    eachClass.append(classTypeList[c])  #i = 1のときclassIDが1のもの（person）クラス
                                                        #として認識された矩形に対応する
                                                        #sortedIoUListがeachClassに格納される

                    #print(classTypeList[c]) 
            if not len(eachClass) == 0:                 #1つでも認識されたクラスであれば
                #print("pre", sum)
                #print("aaa",eachClass)
                presum , presimdiffs = self.det_simdiff(eachClass,[])   #simdiffが最大の組み合わせを得る
                sum += presum           #IoUが最大となる組み合わせを決定する
                simdiffs += presimdiffs 
                #print("tyukan", sum)

        #print("LastSUM", sum)
        #print("LastIoUs", simdiffs) ###おｋ
        #sum:各クラスのIoUの最大の組み合わせと、それに使ったroiの番号の組み合わせ
        

        return sum, simdiffs
    

    def det_simdiff(self, eachClass, simdiffs, sum = 0):#determineIoU:IoUが最大となる組み合わせを決定する
                                                    #eachClass:sortedIoUListの集合
                                                    #IoUs:各roiの最も大きいIoUの値の配列
                                                    # IoUs:[[要素番号，IoU], [要素番号，IoU],...]
        #print("eachClass",eachClass)
        #print("simdiffs", simdiffs)
        if len(eachClass) == 0:
            
            #print("lastIOU",IoUs, sum)
            return sum, simdiffs

        #eachClass(3次元配列)に入っている矩形のIoUの最大値を算出
        max = 0             #最大値
        index = 0           #イテレータ
        i = 0               #最大値を取る値の要素番号
        for each in eachClass:

            #最大値を保有
            if each[0][1] > max:    #each[0][1]:IoU (numAndIoUのIoU)
                max = each[0][1]
                i = index
            index += 1
        
        if max <= 0:                #max = 0なら可算処理自体無駄なので終了
            #print("prcedure end")
            #print("sum", sum)
            return sum, simdiffs


        delete = eachClass.pop(i)
            #最大値として選定された値を保持するsortedIoUListをeachClassから排除
                                            #その中の最大値をIoUsに格納
        #print("max", max)
        #print("del", delete)
        #print("pop", delete[0])             #delete[0]:最大値として選定された値を保持するsortedIoUList
                                            #の中の最大値

        for simdif in simdiffs:
            #print("simdif", simdif)
            if simdif[0][1] == delete[0][0][1]:      #選出された値がすでに選出済みであれば(要素番号で判定)
                                                  #iou:[要素番号，IoU] ,iou[0]:要素番号(i,j)
                                                  #delete[0]:要素番号
                                                  #要素番号が同じ→同じ矩形を参照している→error
                #print("error")
                #print("sum", sum)
                delete.pop(0)
                if not delete == []:
                    eachClass.append(delete)
                return self.det_simdiff(eachClass, simdiffs, sum)    #再帰(重複した値はpopしてある))
        
        #print("delet",delete[0])
        simdiffs.append(delete[0])              #選出済みでなければIoUsに格納
        
        sum += max
        #print("sum", sum)
        return self.det_simdiff(eachClass, simdiffs, sum)


        """

    

    
    def vector_sort(self, vector_list, is_sim):
        #print("ggggggggggggggggggggggggggggggggggggggggg")
        #クラスによらずにそーとしてない？
        #print("vector_list", vector_list)
        
        v_list = []
       
        #minus_list = [] #値がマイナスの(クラス名が異なる)の要素の入れ物

        while not len(vector_list) == 0:
            max = -100
            save = 0
            index = 0

            for i in vector_list:#iは配列
                if max < i[1]:
                    max = i[1]
                    save = index
                index += 1
            
            if max == -1:       #負の値しか残っていなければ
                break
            
            v_list.append(vector_list.pop(save))
            #print(Ilist, save, IoUList)
        #print("v_list", v_list)
        #print("minus_list", minus_list)
        #print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")

        #print("IList",Ilist)
        if not is_sim:          #相違度の計算なら小さい順
            v_list = v_list[::-1] #小さい順にひっくり返す
        #print("vector_list", vector_list)
        while not len(vector_list) == 0:
            v_list.append(vector_list.pop(0))
            #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        
        #v_list += minus_list    #最後に負のものを追加
        #print("v_list", v_list)

        return v_list







    
    def roi_difference(self, roi1, roi2):       #矩形同士の比較
        
        # 類似度は0~1にしたい
        #もしくは相違度1~無限にしたい
        

        #(0，0)から四隅へのベクトルを中心ベクトルの長さで割ったものの外積(差を取らない)
        vector1 = self.vector_zero2roi(roi1)    #np.array[(upper_left, lower_right, center)]
        vector2 = self.vector_zero2roi(roi2) 

        return 
    
    
    def vector_difference(self, vector1, vector2):  #差ベクトルの算出
        #print("差ベクトル開始")
        #vector2 - vector1
        vector1_pos = np.array([vector1[2] - vector1[0], vector1[3] - vector1[1]])  #位置ベクトル

        #print("1_pos_len:", self.vector_len([0, 0, vector1_pos[0], vector1_pos[1]]))
        #print("1_pos_len:", (vector1_pos[0] ** 2 + vector1_pos[1] ** 2) **(1/2))
        
        vector2_pos = np.array([vector2[2] - vector2[0], vector2[3] - vector2[1]])
        #print("2_pos_len:", self.vector_len([0, 0, vector2_pos[0], vector2_pos[1]]))
        #print("2_pos_len:", (vector2_pos[0] ** 2 + vector2_pos[1] ** 2) **(1/2))
        
        #位置ベクトルの差
        vector_diff = np.array([vector2_pos[0], vector2_pos[1], vector1_pos[0], vector1_pos[1]])

        



        
        #print("差ベクトル：",vector_diff)
        return vector_diff
    
    
    def vector_zero2roi(self, roi):
      #2(0,0)から四隅と中心のベクトルを得る((0,0) → roi2)
        upper_left  = np.array([0, 0, roi[0], roi[1] ])
       
        lower_right = np.array([ 0, 0, roi[2], roi[3] ])
        center      = np.array([0, 0, (roi[2] + roi[0]) / 2, (roi[3] + roi[1]) / 2])
        
        return np.array[(upper_left, lower_right, center)]
    
    
    def vector_roi2roi(self, roi1, roi2):
        #2つの矩形から四隅と中心のベクトルを得る(roi1 → roi2のベクトル)

        upper_left  = np.array([roi1[0], roi1[1], roi2[0], roi2[1] ])
       
        lower_right = np.array([ roi1[2], roi1[3], roi2[2], roi2[3] ])

        center      = np.array([(roi1[2] + roi1[0]) / 2, (roi1[3] + roi1[1]) / 2,
                           (roi2[2] + roi2[0]) / 2, (roi2[3] + roi2[1]) / 2  ])
        
        return np.array([upper_left, lower_right, center])

    def vector_center2corner(self, roi):
        #矩形の中心→左上、右下のベクトルを取る

        center_x = (roi[3] + roi[1]) / 2
        center_y = (roi[2] + roi[0]) / 2

        center2TL = np.array([center_y, center_x, roi[0], roi[1]])
        center2BR = np.array([center_y, center_x, roi[2], roi[3]])
        
        vector = [center2TL, center2BR]



        return vector

    def difference_vector_similarity(self, vector1, vector2, diagonal):       #差ベクトルを用いた類似度
        
        vector_diff = self.vector_len(self.vector_difference(vector1, vector2))
        #print("vector_diff", vector_diff)


        #len1 = self.vector_len(vector1)
        #len2 = self.vector_len(vector2)
        #print("len1", len1)
        #print("len2", len2)

        #similarity = 1 - vector_diff / (len1 + len2)
        similarity = 1 - vector_diff / diagonal
        if similarity < 0:
            return 0
        #print("difference_vector_similarity:", similarity)


        return similarity


    def vector_len(self, vector):
        #print("ベクトル長さ")
        #vector = np.array([y0, x0, y1, x1])
        #(x0,y0) → (x1, y1) の長さを求める
        
        x_len = vector[3] - vector[1]
        #print("x_len:{}".format(x_len))

        y_len = vector[2] - vector[0]
        #print("y_len:{}".format(y_len))


        length =  (x_len ** 2 + y_len **2) **(1/2)

        #print("ベクトル長さ:{}".format(length))



        return length


    def vector_cos(self, vector1, vector2 = None ):
        length1 = self.vector_len(vector1) 
        
        if length1 == 0:    #vector１が0ベクトルなら
                print("Cos Can Not Be Computable. length1 = 0")
                #print("vector1",vector1)
                #print("vector2",vector2)
                return None

        #1つだけの場合
        if vector2 == None:
            vector1_x_len = vector1[3] - vector1[1]
            #if vector1_x_len == 0:  #垂直なベクトルなら
            #    return 0   
          
            return vector1_x_len / length1

        #2つの場合
        else:
            print("cos two")
            length2 = self.vector_len(vector2)
            if length2 == 0:    #vector１が0ベクトルなら
                    print("Cos Can Not Be Computable. length2 = 0")
                    #print("vector1",vector1)
                    #print("vector2",vector2)
                    return None
            imp = self.inner_product(vector1, vector2)
        
            cos =  imp / (length1 * length2)

        """
        length1 = self.vector_len(vector1) 
        length2 = self.vector_len(vector2)
        x_len = vector1[3] - vector1[1]     #符号あり
        if x_len == 0:
            #print("x_len = 0")
            return 0

        if length1 == 0:
            #print("Cos Can Not Be Computable.")
            return None

        if length2 == 0:      
            cos =  length1 / x_len
            return cos


        imp = self.inner_product(vector1, vector2)
   
        print("imp", imp)
        cos =  imp / (length1 * length2)
        """

        return cos

    def vector_sin(self, vector1, vector2 = np.array([])):
        #vector2 = None でベクトルと横線のなす角のsinを算出
        #print("sin開始")
        """
        length = self.vector_len(vector)
        y_len = vector[2] - vector[0]
        return = length / y_len
        """
        sin =  (1 - (self.vector_cos(vector1, vector2)) ** 2) ** (1/2) 
        #print("sin終了:{}".format(sin))
        return sin


    def inner_product(self, vector1, vector2):             #内積
        #print("内積開始")
        x_len1 = vector1[3] - vector1[1]    #符号付き長さ
        y_len1 = vector1[2] - vector1[0]
        x_len2 = vector2[3] - vector2[1]
        y_len2 = vector2[2] - vector2[0]

        
        inner_product = x_len1 * x_len2 + y_len1 * y_len2
        #print("内積終了", inner_product)
        return  inner_product
    
    def cross_product_size(self, sizeX, sizeY, vector1, vector2):        #外積の大きさの絶対値
        #print("外積開始")
        #print("cross")
        #print("v1",vector1)
        #print("v2",vector2)
        length1 = self.vector_len(vector1)
        length2 = self.vector_len(vector2)
        sin = self.vector_sin(vector1, vector2)
        cross_product_size = abs(length1 * length2 * sin)
        #print("外積終了:{}".format(cross_product_size))
        max = sizeX * sizeY
        similarity = 1 - cross_product_size / max
        return  similarity

    def cos_by_len(self, vector1, vector2): #|cos|/(|a|*|b|)
        #rint("cos_by_len開始")
        cos = abs(self.vector_cos(vector1, vector2))
        a_length = self.vector_len(vector1)
        b_length = self.vector_len(vector2)

        cos_by_len = abs(cos) / (a_length * b_length)
        cos_by_len = int(10000 * cos_by_len) / 10000
        #print("cos_by_len終了:",cos_by_len)
        return cos_by_len

    def manhattan_dist(self, sizeX, sizeY, vector1, vector2):               #ベクトルからマンハッタン距離を求める
        #print("マンハッタン開始")
        upper_left = (self.vector_len(vector1) ** 2) ** (1/2)       #三平方でマンハッタン距離を求める
        loerw_right = (self.vector_len(vector2) ** 2) ** (1/2)       #三平方でマンハッタン距離を求める

        manhattan = (upper_left + loerw_right) #↾の和

        max = sizeX + sizeY

        simirality = 1 - manhattan / max

        #print("マンハッタン終了", manhattan)

        return simirality

    def add_vector_len(self, sizeX, sizeY, vector1, vector2):
        length1 = self.vector_len(vector1)
        length2 = self.vector_len(vector2)
        add_vector_len = length1 + length2
        dia = math.hypot(sizeX, sizeY)
        #dia = (sizeX ** 2 + sizeY ** 2) ** (1/2)

        #print("dia", dia)
        #print("length1:",length1)
        #print("length2:",length2)
        
        #print("add_vector_len / dia:",add_vector_len / dia)

        simirality = 1 - add_vector_len / (2 * dia)

        if simirality < 0:
            print("dia:", dia)
            print("length1:",length1)
            print("length2:",length2)

        return simirality
    
    def keypoint_similarity(self, keypoint1, keypoint2):
        #これ実行するとそのまま姿勢の類似度が出る
        """
        相対位置リスト
        0.鼻            2：@右肩、@左肩
        1.右目          2：なし
        2.左目          1：なし
        3.右耳          1：なし
        4.左耳           ：なし
        5.右肩          3：鼻、@右ひじ
        6.左肩          2：鼻、@左ひじ
        7.右ひじ        1：右肩、@右手首
        8.左ひじ        1：左肩、@左手首
        9.右手首         ：右ひじ
        10.左手首        ：左ひじ
        11.右尻         2：@右ひざ
        12.左尻         1：@左ひざ
        13.右ひざ       1：右尻、@右くるぶし
        14.左ひざ       1：左尻、@左くるぶし
        15.右くるぶし    ：右ひざ
        16.左くるぶし    ：左ひざ
                       合計：18回
    

        ベクトル = np.array(左上、右上、左下、右下、中心)
        """


        general_keypoint_boolean_list = np.array([
                [0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0]
                ])          #考慮するもの同士は１
        
        #print(general_keypoint_boolean_list)
        #print("keypoint2")
        keypoint_vector_list2, keypoint_boolean_list = \
                        self.keypoint_vector(keypoint2, general_keypoint_boolean_list)
        #print("keypoint1")
        keypoint_vector_list1, keypoint_boolean_list = \
                        self.keypoint_vector(keypoint1, general_keypoint_boolean_list)
        #print("general",general_keypoint_boolean_list)
        #print("bool",keypoint_boolean_list)
        #print("vector1",keypoint_vector_list1)
        #print("vector2",keypoint_vector_list2)
        #keypoint2のkeypoint_boolean_listは使わないので上書きしていい

        similarity = 0
        #全体の類似度の算出
        i = 0
        #print("keypoint",keypoint1, keypoint2)
        #print("bool",keypoint_boolean_list)
        sum_vector_length = 0
        for vectors1, vectors2, booleans in zip(keypoint_vector_list1, 
                                            keypoint_vector_list2, 
                                            keypoint_boolean_list):
                        
                                #listの各行に対して走査
            j = 0
            #print("vectors in 1st for", vectors1, vectors2, booleans)
            #print("i,j",i,j)
            #each_similarity = 0 #行ごとの類似度
            #boolsum_point = 0         #行ごとの考慮すべき点の数
            #sum_vector_length = 0
            for vector1, vector2, boolean in zip(vectors1, vectors2, booleans):
                #print("vector in 2nd for", vector1, vector2, boolean)
             
                if boolean:
                    #print("boolean")
                    #print("i,j",i,j)
                    sim = self.keypoint_vector_similarity(vector1,vector2)
                    vector1_length = self.vector_len(vector1)
                    #each_similarity += sim * vector1_length
                    similarity += sim * vector1_length
                    sum_vector_length += vector1_length
                    #print("each_point similarity",sim)
                    
                    #boolsum_point += 1
                else:
                    #print("not to consider")
                    pass
                j += 1
            #print("boolsum:",boolsum)
            #if not boolsum_point == 0:
            #    similarity += each_similarity / boolsum_point   #考慮したキーポイントの数で割る
                #print("kakunin",each_similarity, boolsum_point)
                #print("each_line similarity", each_similarity / boolsum_point)
            
                
            i += 1

        #if not boolsum_line == 0:
        #    similarity /= boolsum_line
        #print("pose_sim in vector_getter.keypoint_similarity", similarity)

        #print("boollist",keypoint_boolean_list)
        if sum_vector_length == 0:
            similarity = 0
        else:
            similarity /= sum_vector_length

        #print("keypoint_similarity", similarity)
            

        return similarity
    
    def keypoint_vector_similarity(self, vector1, vector2):
        #ベクトルの類似度を計算する
        #vector2が無効の場合がある(0ベクトル)
        #この時、その類似度は0とする
        #vextor１が無効の場合はない(計算した結果0になった場合は除く)
        #これは、vector1を求める段階で0ベクトルのものはkeypoint_boolean_listの対応する値が０と
        #なるようにしているため、そのような場合にこの関数は呼び出されない
        similarity = 0
        #print("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff")


        if not self.vector_len(vector2) == 0:   #vector2が0ベクトルなら計算しない
            if  self.vector_len(vector1) == 0:
                #print("vector1",vector1)
                return similarity

            
            
            #cos1 = (self.vector_cos(vector1) + 1) / 2   #0~1
            #cos2 = (self.vector_cos(vector2) + 1) / 2    #0~1
            #similarity = 1 - abs(cos2 - cos1)   
            cos_sim = self.cos_similarity(vector1, vector2)     #コサイン類似度(-1 ~ 1)
            if cos_sim < 0:
                similarity = 0
            else:            
                similarity = cos_sim

        else:
            if  self.vector_len(vector1) == 0:
                similarity = 1
            else:
                similarity = 0 #描かなくてもいい
            #print("vector1",vector1)
            #print("vector2",vector2)
        
     

        return similarity
    def cos_similarity(self, vector1, vector2):
        #ベクトルとx軸のなす角(rad)を求める
        inner_product = self.inner_product(vector1, vector2)

        len1 = abs(self.vector_len(vector1)) 

        len2 = abs(self.vector_len(vector2))


        sim = inner_product / (len1 * len2)

     

        return sim

    def keypoint_vector(self, keypoint, general_keypoint_boolean_list):
        """
        相対位置リスト
        0.鼻            2：@右肩5、@左肩6
        1.右目          2：@左目2、@右耳3
        2.左目          1：右目1、@左耳4
        3.右耳          1：@左耳4、右目1
        4.左耳           ：右耳3、左目2
        5.右肩          3：鼻0、@左肩6、@右ひじ7、@右尻11
        6.左肩          2：鼻0、右肩5、@左ひじ8、@左尻12
        7.右ひじ        1：右肩5、@右手首9
        8.左ひじ        1：左肩6、@左手首10
        9.右手首         ：右ひじ7
        10.左手首        ：左ひじ8
        11.右尻         2：@左尻12、右肩5、@右ひざ13
        12.左尻         1：右尻11、左肩6、@左ひざ14
        13.右ひざ       1：右尻11、@右くるぶし15
        14.左ひざ       1：左尻12、@左くるぶし16
        15.右くるぶし    ：右ひざ13
        16.左くるぶし    ：左ひざ14
                       合計：18回
    
        ベクトル = ndarray([左上y、左上ｘ、左下ｙ、右下ｘ])
        keypoint :     姿勢
        keypoint[0]    ：各キーポイント[x,y,bool]

        """
    
        keypoint_boolean_list = copy.copy(general_keypoint_boolean_list)

        keypoint_vector_list = np.zeros((17,17,4))  #縦、横、(y0,x0,y1,x1)
     
        #print("for start")
        for i in range(0, keypoint_boolean_list.shape[0]):   #縦(終点)
            vectors = np.zeros((17,4))
            for j in range(0, general_keypoint_boolean_list.shape[1]):#横(始点)
                vector = np.zeros(4)
                    #vector:キーポイント[i]から[j]のベクトル
                #print(i,j)
                if keypoint_boolean_list[i][j] == 1:    #考慮するべき点であれば
                    #print("to_consider")
                    #print("keypoint",keypoint)
                    if keypoint[i][2] == 1 and keypoint[j][2] == 1:#(x,y,bool)のbool == trueであれば     
                        #print("bool_true")
                        #上ndarrayのindex out of bounds exception
                        vector = np.array([keypoint[j][1], keypoint[j][0],
                                        keypoint[i][1], keypoint[i][0]])
                    else:
                        #print("a", keypoint_boolean_list[i][j], i, j)
                        keypoint_boolean_list[i][j] = 0
                        
                        #vector = np.array([keypoint[j][1], keypoint[j][0],
                        #                keypoint[i][1], keypoint[i][0]])
                                #考慮すべき点が見つからなければ
                                #→booleanlist事態を書き換える
                        #print("bool_false")
                else:
                    #print("not to consider")
                    #keypoint_boolean_list[i][j] == 0
                    pass
                #print("vector", vector)

                vectors[j] = vector

            keypoint_vector_list[i] = vectors 
            
        #print("keypoint_vector_list",keypoint_vector_list)
        #print("end_of_func bool", keypoint_boolean_list)

        return keypoint_vector_list, keypoint_boolean_list

    def reverse_vector(self, vector):       #逆向きベクトル

        return np.array([vector[2],vector[3],vector[0],vector[1]])
    
  

def main():


    return

main()

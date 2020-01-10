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

from vector_getter import  Get_vector

import copy
import traceback
# Import COCO config
#import coco




class Get_IoU:
    def __init__(self, class_names):
        self.class_names = class_names
        self.vector_computer = Get_vector()
        return



    def center_sim(self, center2corner1, center2corner2, diagonal):
        upper_sim = self.vector_computer.difference_vector_similarity(center2corner1[0], center2corner2[0], diagonal)
        lower_sim = self.vector_computer.difference_vector_similarity(center2corner1[1], center2corner2[1], diagonal)
        center_similarity = (upper_sim + lower_sim) / 2 


        return center_similarity

    def det_with_comparison_roi(self, data1, data2):
        #中心矩形の決定,  対応矩形と類似度の決定

        #対角線を計算
        diagonal = (data1.sizeX ** 2 + data1.sizeY ** 2) ** (1 / 2)
        

        if data2.roi.shape[0] == 1: #画像の矩形が1つだったら
            center_roi_number2 = 0          #画像の中心矩形確定
            max_correspondence = []
            similarity_list = []
            max_similarity = -1
            max_index = -1
            i = 0
            center2corner2 = self.vector_computer.vector_center2corner(data2.roi[center_roi_number2])
            #構図データの中心矩形を順に中心矩形候補として扱い，中心矩形の計算法で類似度を求める
            for center_roi_number1 in range(data1.roi.shape[0]): 
                if data1.classID[center_roi_number1] == data2.classID[center_roi_number2]:  #クラス名が等しい
                    
                    center2corner1 = self.vector_computer.vector_center2corner(data1.roi[center_roi_number1])
                    #中心矩形同士の類似度を計算
                    center_similarity = self.center_sim(center2corner1, center2corner2, diagonal)
                else:   #クラス名が異なる
                    center_similarity = -1
                column_sim = [center_similarity]
                similarity_list.append(column_sim)  #縦に類似度を入れていく

                if center_similarity > max_similarity:      #類似度が最大になったら
                    max_similarity = center_similarity
                    max_index = i
                i += 1
            
            if not max_index == -1: #対応するクラスがあったら
                center_roi_number1 = max_index
                max_correspondence = [[center_roi_number1, center_roi_number2]]
            
            
            center_roi_numbers = {"1": center_roi_number1, "2": center_roi_number2}
            

            return similarity_list, max_correspondence, center_roi_numbers

        elif data1.roi.shape[0] == 1: #構図データの矩形が1つしかなかったら
            #画像の矩形は複数ある(上で画像の矩形が1つの場合の処理は終わってるため)
            center_roi_number1 = 0
            max_correspondence = []
            similarity_list = []
            sim_list = []
            max_similarity = -1
            max_index = -1
            j = 0
            center2corner1 = self.vector_computer.vector_center2corner(data1.roi[center_roi_number1])
            
            #画像の矩形を順に中心矩形候補として扱い，中心矩形の計算法で求める
            for center_roi_number2 in range(data2.roi.shape[0]): 
                if data1.classID[center_roi_number1] == data2.classID[center_roi_number2]:  #クラス名が等しい
                    
                    
                    center2corner2 = self.vector_computer.vector_center2corner(data2.roi[center_roi_number2])
                    #中心矩形同士の類似度を計算
                    center_similarity = self.center_sim(center2corner1, center2corner2, diagonal)

                else:   #クラス名が異なる
                    center_similarity = -1
                sim_list.append(center_similarity)  #類似度を入れていく
                

                if center_similarity > max_similarity:      #類似度が最大になったら
                    max_similarity = center_similarity
                    max_index = j
                j += 1

            similarity_list.append(sim_list)
            
            if not max_index == -1: #対応するクラスと類似度を計算出来たら
                center_roi_number2 = max_index
                max_correspondence = [[center_roi_number1, center_roi_number2]]
            else:
                center_roi_number2 = None

            center_roi_numbers = {"1": center_roi_number1, "2": center_roi_number2}
            

            return similarity_list, max_correspondence, center_roi_numbers


        else:   #構図データ，画像ともに複数の矩形を持つ
        
            #data1(構図データ)のroiの重心位置
            center_x, center_y = self.det_center_of_gravity(data1)  
    
            #中心矩形１の番号
            center_roi_number1 = self.det_center_roi(data1, center_x, center_y, data2) 

            if center_roi_number1 == -1:    #data1とdata2で重複するクラスがない
                similarity_list = []
                max_correspondence = []
                center_roi_numbers = {"1": None, "2": None}
                return similarity_list, max_correspondence, center_roi_numbers
            
            #data1の中心矩形からの各矩形のベクトルを算出
            data1_vectors = self.roi_relative_point(data1, center_roi_number1)
                                    #data1_sim:[[upper_lower_cos],[upper_lower_cos],...]
            #print("data1.vectors:", data1_vectors)

            max_sim = 0
            center_roi_number2 = -1
            max_correspondence = []
            similarity_list = []
            #data2の矩形を順に中心矩形候補としてベクトルを算出，data1との対応矩形と類似度を計算，
            for center_roi_number in range(data2.roi.shape[0]):
                if data1.classID[center_roi_number1] == data2.classID[center_roi_number]:
                    #ベクトルを計算
                    data2_vectors = self.roi_relative_point(data2, center_roi_number)
                                        #data2_sim:[[upper_lower_cos],[upper_lower_cos],...]
                    
                    #類似度と対応を計算
                    #print("center_roi_number2:",center_roi_number)
                    sim, correspondence, sim_list = self.det_roi_correspondence(data1_vectors, data2_vectors,    
                                                                    data1.classID, data2.classID, diagonal)
            
                    #矩形の数で割って正規化
                    sim /= data1.roi.shape[0]
                    
                    if sim > max_sim:
                        max_sim = sim
                        center_roi_number2 = center_roi_number
                        max_correspondence = correspondence
                        similarity_list = sim_list
                        #print("center_roi_number in max_sim:", center_roi_number2)
                        #print("corres in max_sim", max_correspondence)
                        #print("sim_list in max_sim:\n", similarity_list)

            #print("corres in 167", max_correspondence)
            #print("sim_list in 168:\n", similarity_list)
        

    
            #中心矩形の中心→角のベクトルを計算
            center2corner1 = self.vector_computer.vector_center2corner(data1.roi[center_roi_number1])
            center2corner2 = self.vector_computer.vector_center2corner(data2.roi[center_roi_number2])
            #中心矩形同士の類似度を計算
            center_similarity = self.center_sim(center2corner1, center2corner2, diagonal)

            #print("center_sim", center_similarity)
            if center_similarity > 1:
                print("over center_sim", center_similarity)
            elif center_similarity < 0:
                print("under center_sim", center_similarity)
            #中心矩形同士の対応矩形の類似度を書き換える
            similarity_list[center_roi_number1][center_roi_number2] = center_similarity
            #print("center_index:",center_roi_number1, center_roi_number2)
            
            
            center_roi_numbers = {"1": center_roi_number1, "2": center_roi_number2}


            return similarity_list, max_correspondence, center_roi_numbers


    def det_roi_correspondence(self, data1_vectors, data2_vectors, classID1, classID2, diagonal):
        #data1_vectors:[[upper_lower_vector],[upper_lower_vector],...]
        #data2_vectors:[[upper_lower_vrctor],[upper_lower_vector],...]
        #クラス名を考慮する
        #print("ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
        

        #矩形間のの類似度リスト(総当たり)
        similarity_list = np.zeros((classID1.shape[0], classID2.shape[0]))
        
        i = 0
        for vector1 ,id1 in zip(data1_vectors, classID1):
            j = 0
            for vector2, id2 in zip(data2_vectors, classID2):
                #print("i,j", i,j)
                #print("id:",id1, id2)
                if id1 == id2:  #クラス名が等しい → 差ベクトルで類似度計算
                    #print("ffffffffffffffffffff")
                    #左上ベクトル同士の類似度
                    #print("vectors1", data1_vectors)
                    #print("vectors2", data2_vectors)
                    upper_sim = self.vector_computer.difference_vector_similarity(vector1[0], vector2[0], diagonal)
                    #print("upper_sim", upper_sim)
                    
                    if upper_sim > 1:
                        print("over upper_sim  230:", upper_sim)
                    elif upper_sim < 0:
                        print("under upper_sim 232:", upper_sim)
                    
                    #右下ベクトル同士の類似度
                    lower_sim = self.vector_computer.difference_vector_similarity(vector1[1], vector2[1], diagonal)
                    
                    if lower_sim > 1:
                        print("over lower_sim 238:", lower_sim)
                    elif lower_sim < 0:
                        print("under lower_sim 240:", lower_sim)
                    
                    #最終的な類似度
                    similarity_list[i][j] = (upper_sim + lower_sim) / 2
                    
                else:
                    #クラス名が等しくない
                    #print("aaaaaaa")
                    similarity_list[i][j] = -1
                j += 1
            i += 1

        #similatiry_list完成

        #矩形の対応を決定，correspondenceに格納，sim_sumに類似度の合計を格納
        #print("sim_list:\n", similarity_list)
        sim_sum, correspondence = self.detIoU2_keypoint(copy.copy(similarity_list))
        return sim_sum, correspondence, similarity_list

    
    def roi_relative_point(self, data, center_roi_number):
        #中心矩形からの各矩形までのベクトルを求める
        #中心矩形の中心から、各矩形の左上、右下までのベクトルを取る

        vectors = []
        #print("data.roi:", data.roi)
        #print("[center_roi_number]:",center_roi_number)
        center_roi = data.roi[center_roi_number]
        center_x = (center_roi[1] + center_roi[3]) / 2
        center_y = (center_roi[0] + center_roi[2]) / 2
        count = 0
        for roi in data.roi:
            upper_vector = np.array([center_y, center_x , roi[0], roi[1]])
            lower_vector = np.array([center_y, center_x , roi[2], roi[3]])

            upper_lower_vector = [upper_vector, lower_vector]
            vectors.append(upper_lower_vector)


        return  vectors    #vectors[[upper_lower_vector],[upper_lower_vector],...]
                        #[upper_lower_vector] = [upper_vector, lower_vector]

    def det_center_roi(self, data, center_of_gravity_x, center_of_gravity_y, data2):
        #重心から最も近い重心をもつ矩形を中心矩形とする
        #中心矩形のクラスはdata2に検出されているものに限る





        nearest_similarity = 0 
        nearest_roi_number = -1
        max_distance = (data.sizeX ** 2 +  data.sizeY ** 2) ** (1/2)
                    #重心との最大距離(画像の対角線の長さの半分)
        i = 0
        for roi, classID in zip(data.roi, data.classID):
            if not classID in data2.classID:    #data2に含まれるクラスでない
                i += 1
                continue

            roi_center_x = (roi[3] + roi[1]) / 2
            roi_center_y = (roi[2] + roi[0]) / 2

            distance_x = abs(center_of_gravity_x - roi_center_x)
            distance_y = abs(center_of_gravity_y - roi_center_y)
            distance = (distance_x ** 2 + distance_y ** 2) ** (1/2) #矩形と重心の距離
            area = (roi[3] - roi[1]) * (roi[2] - roi[0])

            similarity = (1 - distance / max_distance) * area
            
            if nearest_similarity < similarity:
                nearest_similarity = similarity
                nearest_roi_number = i

            i += 1

        return nearest_roi_number
    
    def det_center_of_gravity(self, data):
        #重心を求める
            #矩形の中点に長さに比例した重みを与え、合計して画像の長さで割る
        
        roi_area_sum_x = 0    #重み付き長さ
        roi_area_sum_y = 0    #の合計

        length_sum_x = 0    #長さの合計
        length_sum_y = 0



        for roi in data.roi:
            roi_x_len = roi[3] - roi[1]                 #矩形の長さ(x)→各矩形にかかる重みと等しい
            roi_x_center =  (roi[3] + roi[1]) / 2       #矩形の中点(ⅹ)→各矩形の重心位置
            roi_area_sum_x += roi_x_center * roi_x_len  #重みつき重心
            length_sum_x += roi_x_len

            
            roi_y_len = roi[2] - roi[0]
            roi_y_center =  (roi[2] + roi[0]) / 2
            roi_area_sum_y += roi_y_center * roi_y_len
            length_sum_y += roi_y_len
        
        center_x = roi_area_sum_x / length_sum_x
        center_y = roi_area_sum_y / length_sum_y

        return center_x, center_y


    def area_normalize(self, data1, data2): 
        #data2をdata1の面積に合わせる


        #変換倍率
        ratio = (data1.sizeX * data1.sizeY) / (data2.sizeX * data2.sizeY)

        normalized_data2 = copy.copy(data2)

        #サイズを正規化
        normalized_data2.sizeX *= ratio
        normalized_data2.sizeY *= ratio

        #矩形サイズを正規化
        for roi in normalized_data2.roi:
            roi[0] *= ratio
            roi[1] *= ratio
            roi[2] *= ratio
            roi[3] *= ratio


        return normalized_data2

    def inform_invalid_value(self, data1, data2, messages, similarity_list = [],
                            correspondence = [], center_roi_numbers = []):
        #矩形の類似度がおかしいときに種々のデータを表示
        #keypoint_resultの各値(名前込み)
        #矩形の類似度リスト
        #対応矩形リスト
        #中心矩形番号

        print("\n###########################inform_invalid_velue################################")
        for message in messages:
            print("\n{}".format(message))
        print("\nkouzu_data file_name:", data1.file_name)
        print("image_data file_name:", data2.file_name)
        print("kouzu_data class:", data1.classID)
        print("image_data class:", data2.classID)
        print("kouzu_data roi:\n", data1.roi)
        print("image_data roi:\n", data2.roi)


        print("similarity_list:\n", similarity_list)
        print("correspondence:", correspondence)
        print("center_roi_numbers:", center_roi_numbers)

        print("############################close inform_invalid_velue.#########################\n")
        if len(messages) > 0:
            sys.exit()


        return

    def is_invalid_value(self, data1, data2, similarity_list, correspondence, center_roi_numbers):
        messages = []
        i = 0
        
        #print("len similarity_list", len(similarity_list))
        for sim_list in similarity_list:
            j = 0
            #print("len sim_list", len(sim_list))
            for sim in sim_list:
                
                if sim < 0 and data1.classID[i] == data2.classID[j]:
                    messages.append("similarity under zero. index:{}".format([i, j]))
                elif sim > 1:
                    messages.append("similarity over one. index:{}".format([i, j]))
                
                j += 1
            i += 1

        
        if len(messages) > 0:
            self.inform_invalid_value(data1, data2, messages = messages,
                                    similarity_list = similarity_list,
                                    correspondence = correspondence,
                                    center_roi_numbers = center_roi_numbers)
            
            return True
        

        return False

            
            
        
        

    def get_similarity(self, data1, data2, should_best_weight_compute, keypoint_weight, is_disclete_sim):      #比較実行関数
        """
        data1：構図データ
        data2：画像
        should_best_weight_compute：キーポイントの重みの最良結果を求めるべきか
        keypoint_weight：すでに決まっているキーポイントの重みがあれば入力
        is_disclete_sim:キーポイントと矩形の類似度を個別で返すかどうか
        """
        
        #正規化(縦横比を変更せずに面積をそろえる)
        normalized_data2= self.area_normalize(copy.copy(data1), copy.copy(data2)) 
        #print("data2_class", normalized_data2.classID)
        #print("data1_class", data1.classID)


        #矩形の類似度決定
        try:
            roi_similarity_list, correspondence, center_roi_numbers = self.det_with_comparison_roi(data1, normalized_data2)
        except Exception as ex:
            #エラー時のデータ表示]

            messages = [traceback.format_exc(), "exception in det_with_comparison_roi()"]
            self.inform_invalid_value(data1, data2, messages)


            return None, None

        self.is_invalid_value(data1, data2, similarity_list = roi_similarity_list,
                                                correspondence = correspondence,
                                                center_roi_numbers = center_roi_numbers)

                                                
        #self.inform_invalid_value(data1, data2, messages = [], 
        #                            similarity_list = roi_similarity_list,
        #                            correspondence = correspondence,
        #                            center_roi_numbers = center_roi_numbers)
        

        keypoint1 = data1.keypoint
        keypoint2 = data2.keypoint
        num_sim_is_keypoints = []
        #構図データ内の全矩形の面積の合計
        data1_roi_area_sum = 0
        #構図データ内のキーポイントを持つ矩形の面積の合計
        data1_keypoint_area_sum = 0

        for roi, pose in zip(data1.roi, data1.keypoint):
            #構図データのi番目の矩形の面積
            area_size = (roi[2] - roi[0]) * (roi[3] - roi[1])
            data1_roi_area_sum += area_size

            for point in pose:
                if point[-1] == 1:
                    #キーポイントが少しでも入力されていればキーポイントの重みを入力
                    data1_keypoint_area_sum += area_size   
                    break
           

        #対応する番号同士の類似度を算出
        for corres in correspondence:
            #print("corres", corres)

            i = corres[0]
            j = corres[1]
            
            #構図データのi番目の矩形の面積
            area_size = (data1.roi[i][2] - data1.roi[i][0]) * (data1.roi[i][3] - data1.roi[i][1])
            #data1_roi_area_sum += area_size
            
            #対応する矩形の番号同士を格納, #矩形の類似度を格納
            #print("i,j", i,j, roi_similarity_list)
            num_sim_is_keypoint = {
                                    "correspond_number" : {"data1":i, "data2":j},
                                    "roi_sim" : roi_similarity_list[i][j],
                                    "area_size" : area_size
                                    }                   

            #クラス名が人
            if data1.classID[i] == 1:
                #対応矩形内のキーポイントの類似度
                num_sim_is_keypoint["keypoint_sim"] = self.vector_computer.keypoint_similarity(
                                        keypoint1[i], keypoint2[j])                            
            else:
                #人以外なら姿勢類似度は無視
                num_sim_is_keypoint["keypoint_sim"] = num_sim_is_keypoint["roi_sim"] 
            
            #キーポイントが入力されているかどうか
            num_sim_is_keypoint["is_keypoint"] = False
            for point in keypoint1[i]:
                if point[-1] == 1:
                    #キーポイントが少しでも入力されていればTrue
                    num_sim_is_keypoint["is_keypoint"] =  True       
                    break
            
            """
            #キーポイントの類似度の値域判定
            if 1 < keypoint_similarity:
                print("over_key!", keypoint_similarity)
            elif keypoint_similarity < 0:
                print("under_key", keypoint_similarity)
            if 1 < roi_similarity:
                print("over_roi!", roi_similarity)
            elif roi_similarity < 0:
                print("under_roi", roi_similarity)
            """
            #最後に配列に格納
            num_sim_is_keypoints.append(num_sim_is_keypoint)

        #最良の重みを決定する処理
        if should_best_weight_compute:
            total_similarity, best_keypoint_weight = self.det_best_weight(num_sim_is_keypoints, 
                                                                        data1_roi_area_sum,
                                                                        data1_keypoint_area_sum)

            return total_similarity, best_keypoint_weight

        else:
            
            #最終的な類似度を決定
            if is_disclete_sim:
                roi_similarity, keypoint_similarity = self.roi_keypoint_similarity(num_sim_is_keypoints,  
                                                                            data1_roi_area_sum, 
                                                                            data1_keypoint_area_sum)
                                                                            
                return roi_similarity, keypoint_similarity


            total_similarity = self.det_total_similarity(num_sim_is_keypoints,
                                                         keypoint_weight,
                                                         data1_roi_area_sum,
                                                         data1_keypoint_area_sum)

            return total_similarity, keypoint_weight
    
    def roi_keypoint_similarity(self, num_sim_is_keypoints, data1_roi_area_sum, data1_keypoint_area_sum):
        roi_similarity = 0
        keypoint_similarity = 0
        for num_sim_is_keypoint in num_sim_is_keypoints:
            roi_similarity += num_sim_is_keypoint["roi_sim"] * num_sim_is_keypoint["area_size"]
            if num_sim_is_keypoint["is_keypoint"]:
                keypoint_similarity += num_sim_is_keypoint["keypoint_sim"] * num_sim_is_keypoint["area_size"]


        roi_similarity /= data1_roi_area_sum
        keypoint_similarity /= data1_keypoint_area_sum

        #print("roi_similarity:",roi_similarity)
        #print("keypoint_similarity:",keypoint_similarity)
        return roi_similarity, keypoint_similarity


    
    def det_best_weight(self, num_sim_is_keypoints, data1_roi_area_sum, data1_keypoint_area_sum):
        #10^(-n)刻みの探索で最良の重みを決定する
        ratio = 0.1         #最初0.1刻み
        weight = 0.5  #0.5から前後に0.1刻み
        total_similarity = 0
        keypoint_weight = -1

        while ratio >= 0.01:
            if  weight >= ratio * 5: 
                weight -= ratio * 5
            for i in range(0, 10):   #10回繰り返す
                sim = self.det_total_similarity(num_sim_is_keypoints, weight, 
                                                    data1_roi_area_sum, data1_keypoint_area_sum)
                #print("sim:", sim)
                #print("weight", weight)
                if total_similarity < sim:
                    total_similarity = sim
                    keypoint_weight = weight
                weight += ratio
            
            ratio /= 10
            weight = keypoint_weight
            
        return total_similarity, keypoint_weight
    
    def det_total_similarity(self, num_sim_is_keypoints, keypoint_weight, data1_roi_area_sum, data1_keypoint_area_sum):
        """
        roi_similarity = 0
        keypoint_similarity = 0
        for num_sim_is_keypoint in num_sim_is_keypoints:
            roi_similarity += num_sim_is_keypoint["roi_sim"] * num_sim_is_keypoint["area_size"]

            if num_sim_is_keypoint["is_keypoint"]:
                keypoint_similarity += num_sim_is_keypoint["keypoint_sim"] * num_sim_is_keypoint["area_size"]


        roi_similarity /= data1_roi_area_sum
        keypoint_similarity /= data1_keypoint_area_sum

        #if data1_roi_area_sum == 0: return 0

        total_similarity = keypoint_similarity * keypoint_weight + \
                            roi_similarity * (1 - keypoint_weight)
        #print("\nweight:", keypoint_weight)
        #print("roi_similarity:",roi_similarity)
        #print("keypoint_similarity:",keypoint_similarity)
        #print("total_similarity:",total_similarity)
        return total_similarity
        """

        roi_similarity = 0
        keypoint_similarity = 0
        for num_sim_is_keypoint in num_sim_is_keypoints:
            roi_similarity += num_sim_is_keypoint["roi_sim"] * num_sim_is_keypoint["area_size"]

            keypoint_similarity += num_sim_is_keypoint["keypoint_sim"] * num_sim_is_keypoint["area_size"]


        roi_similarity /= data1_roi_area_sum
        keypoint_similarity /= data1_roi_area_sum

        #if data1_roi_area_sum == 0: return 0

        total_similarity = keypoint_similarity * keypoint_weight + \
                            roi_similarity * (1 - keypoint_weight)

        print("\nweight:", keypoint_weight)
        print("roi_similarity:",roi_similarity)
        print("keypoint_similarity:",keypoint_similarity)
        print("total_similarity:",total_similarity)
        return total_similarity
        


    def keypoint_roi_similarity2(self, keypoint_similarity,roi_similarity, keypoint):
        
        #similarity = keypoint_similarity * keypoint_similarity + \
        #            (1 - keypoint_similarity) * roi_similarity
        #キーポイントの類似度:矩形の類似度 = キーポイントの類似度：(1-キーポイントの類似度)
        is_keypoint_effective = False
        keypoint_weight = 0.7

        #print("similarity in keypoint_roi_similarity k:r", keypoint_similarity, roi_similarity )
        
        for point in keypoint:
            if point[-1] == 1:
                is_keypoint_effective == True       #キーポイントが少しでも入力されていれば
                similarity = keypoint_similarity * keypoint_weight + roi_similarity * (1 - keypoint_weight)
                return similarity, is_keypoint_effective

        similarity = roi_similarity

        



        return similarity, is_keypoint_effective

    def detIoU2_keypoint(self, similarity_list):

        """
        similarity_list:          
                [
                    [[[0, 0], 0.01], [[0, 1], 0.02], [[0, 2], 0.02]]
                    [[[1, 0], 0.01], [[1, 1], 0.02], [[1, 2], 0.02]]
                    [[[2, 0], 0.01], [[2, 1], 0.02], [[2, 2], 0.02]]
                ]
        similarity_list[0]:       [[[0, 0], 0.01], [[0, 1], 0.02], [[0, 2], 0.02]]
        similarity_list[0][0]:    [[0, 0], 0.01]
        similarity_list[0][0][0]: [0, 0]
        similarity_list[0][0][1]: 0.01
        """

        sum = 0
        correspondence = []
        #count = 0
        while True:
            #print("while...")
            #print(similarity_list)
            max = -1     #eachClass内の類似度の最大
            col = 0
            lin = 0

            i = 0
            for each in similarity_list:
                #print("each",each)
                #each =　similarity_list[0]:  [0.01, 0.02, 0.02]
                j = 0
                for e in each:
                    #print(i,j)
                    #print("e:",e)
                    #e = 0.01
                    #print("e",e)
                    if e > max:
                        max = e
                        col = j   #列を代入
                        lin = i   #行を代入
                        #print("collin", col, lin)
                        #print("max,col,lin",max,col,lin)
                    j += 1
                i += 1

            if max < 0:
                #print("no max")
                break

            #print("pre_append eachClass", similarity_list)
            #print("col", col)
            #print("lin", lin)

            correspondence.append([lin, col])
            sum += max

            
            similarity_list[lin] = -1
            similarity_list[:,[col]] = -1

        #返り値：類似度、対応表

        
        return sum, correspondence
     



    def isIoUComputable(self, ratio1, ratio2):
        x1 = ratio1[3] - ratio1[1]
        y1 = ratio1[2] - ratio1[0]
        x2 = ratio2[3] - ratio2[1]
        y2 = ratio2[2] - ratio2[0]

        if ratio1[3] <= ratio2[1]:
            #print("x:1 < 2")
            return False
        elif ratio2[3] <= ratio1[1]:
            #print("x:2 < 1")
            return False
        elif ratio1[2] <= ratio2[0]:
            #print("y:1 < 2")
            return False
        elif ratio2[2] <= ratio1[0]:
            #print("y:2 < 1")
            return False 


        #print("x,y:ok")
        return True

    def iou(self, ratio1, ratio2):  #2つの矩形のIoUの算出
        """
        ratio1[左上Y 左上X 右下Y 右下X]

        ratio2[左上Y 左上X 右下Y 右下X]
        """
        
        ##############koko
        andArea = self.detAnd(ratio1, ratio2)
        #print("SSSSSSSSSSandArea", andArea)
        if len(andArea) == 0:
            return 0

        andValue = (andArea[2] - andArea[0]) * (andArea[3] - andArea[1])
        #print(andArea[2] - andArea[0] , andArea[3] - andArea[1])
        #print(andArea[2], andArea[0], andArea[3], andArea[1])

            
        ratio1Value = (ratio1[2] - ratio1[0]) * \
                            (ratio1[3] - ratio1[1])
        
        #print(ratio1[0][2] - ratio1[0][0], ratio1[0][3] - ratio1[0][1])
        #print(ratio1[0][2] , ratio1[0][0], ratio1[0][3] , ratio1[0][1])

        ratio2Value = (ratio2[2] - ratio2[0]) * \
                            (ratio2[3] - ratio2[1])

        #print(ratio2[0][2] - ratio2[0][0], ratio2[0][3] - ratio2[0][1])
        #print(ratio2[0][2] , ratio2[0][0], ratio2[0][3] , ratio2[0][1])
            
        orValue = ratio1Value + ratio2Value - andValue

        IoU = int(andValue / orValue * 100)
        IoU /= 100
        """
        print("andArea",andArea)
        print("andValue", andValue)
        print("1Value", ratio1Value)
        print("2value", ratio2Value)
        print("or", orValue)
        print("IoU",IoU)
        """

        return IoU

    def detAnd(self, ratio1, ratio2):
        if not self.isIoUComputable(ratio1, ratio2):
            #print("invalid IOU")
            return []
        
        #矩形が交わる座標を求める
        andArea = []
        count = 0
        for r1, r2 in zip(ratio1, ratio2):  #検出数が等しい前提
            if count <= 1:
                if r1 > r2:
                    andArea.append(r1)
                    #print("u",r1, r2)
                else:
                    andArea.append(r2)
                    #print("l",r1, r2)
            else:
                if r1 < r2:
                    andArea.append(r1)
                    #print("u",r1, r2)
                else:
                    andArea.append(r2)
                    #print("l",r1, r2)

            count += 1

        #print("AAAAAAAAAAndArea", andArea)

        return andArea

    

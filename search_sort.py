import os
import sys
import time
import random
import math

import traceback
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

import copy

import loadAndComparison as Lac
from loadAndComparison import ImageData
from loadAndComparison import LoadAndComparison
from loadAndComparison import Keypoint_result

from visualize_keypoint import Visualize_IoU
import glob

"""
画像を検索してソートする。

visualize_IoUに順次pickleデータを入れていく
帰ってきた類似度の値を配列かなんかに画像の名前と一緒に保持

その配列を類似度の大きい順にソートしてその順番に表示(pltで)
↾スクロールできるといいね！！！
↾一緒に類似度も表示できるといいね！！！

以上


"""

class Search_sort():
    def __init__(self, search_image_name):
        """
        search_image_name:探している画像(pickle)
        これに基づいてtokenを生成する
        """
        self.lac = LoadAndComparison()
    
        self.imageDataToken = self.roi_into_image(
                self.lac.load_kouzu_data(search_image_name, is_display=False))
        
        return
    
    def roi_into_image(self, keypoint_result):

        for roi in keypoint_result.roi:
            if roi[0] < 0:
                roi[0] = 0
            
            if roi[1] < 0:
                roi[1] = 0

            if roi[2] > keypoint_result.sizeY - 1:
                roi[2] = keypoint_result.sizeY - 1
            
            if roi[3] > keypoint_result.sizeX - 1:
                roi[3] = keypoint_result.sizeX - 1
            





        return keypoint_result
    
   

    def get_similarity_fromAllpickle_keypoint(self, keypoint_weight):
        """
        pickleフォルダ内のデータをすべて読み込み
        pickles[]に入れる
        各々のpickleデータとtokenImageを比較して類似度を出す

        """
        #, illust , similarity_type
        pickles = []                #画像データと類似度の入れ物
       
        os.chdir('./pickle_keypoint')
        #path = os.getcwd()
        #print("aaaaaaaaaaaaaaaaaa", path)
        file_names = glob.glob("*.pickle")
        os.chdir('../')
        #print(file_names)
        count = 0
        for file_name in file_names:
            pickle_similarity = []
            keypoint_result = self.lac.load_keypoint(keypoint_pickle_name = file_name, 
                                            is_display = False)
            #pickle_similarity.append(keypoint_result)
            pickle_similarity.append(keypoint_result.file_name)
            try:
                similarity, weight = self.lac.similarity_difference(
                                data1 = self.imageDataToken, 
                                data2 = keypoint_result,
                                should_best_weight_compute = False, 
                                keypoint_weight = keypoint_weight)
            except:
               
                traceback.print_exc()
                a = input()

                print("no", count)
                similarity = 0
            pickle_similarity.append(similarity)
            #############################################data1に構図データ、data2にデータベースの画像群#######################################################################################################

            pickles.append(pickle_similarity)
            #print(pickle_similarity[0].file_name)
            #print(pickle_similarity[1][0])
            count += 1
            



        """
        for pickle in pickles:
            #print("a")
            print(pickle[0].file_name, pickle[1][0])
        """ 


        return pickles
    
    def sort(self, keypoint_weight, answer_image_name):
        pickles = self.get_similarity_fromAllpickle_keypoint(keypoint_weight = keypoint_weight)
        #pickles:[[imageData,[sim, deff]],[imageData,[sim, deff]]]
        #これを類似度順にソートする
        #print("merge START")
        pickles = self.merge_sort(pickles)

        
        count = 0
        
        for pickle in pickles:
            if answer_image_name == pickle[0]:
                print(count, pickle[0], pickle[1])
            count += 1
        print("1:", pickles[0][0], pickles[0][1])
        
    
        return pickles


    def merge_sort(self, arr):
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        # ここで分割を行う
        left = arr[:mid]
        right = arr[mid:]

        # 再帰的に分割を行う
        left = self.merge_sort(left)
        right = self.merge_sort(right)

        # returnが返ってきたら、結合を行い、結合したものを次に渡す
        return self.merge(left, right)


    def merge(self, left, right):
        merged = []
        l_i, r_i = 0, 0

        # ソート済み配列をマージするため、それぞれ左から見ていくだけで良い
        while l_i < len(left) and r_i < len(right):
            # ここで=をつけることで安定性を保っている
            if left[l_i][1] >= right[r_i][1]:

                merged.append(left[l_i])
                l_i += 1
            else:
                merged.append(right[r_i])
                r_i += 1

        # 上のwhile文のどちらかがFalseになった場合終了するため、あまりをextendする
        if l_i < len(left):
            merged.extend(left[l_i:])
        if r_i < len(right):
            merged.extend(right[r_i:])
        return merged
    

    def get_best_keypoint_weight(self, answer_image_name):
        #正解画像と読み込んだ構図データの類似度が最大になるように重みを決定する
        answer_image = self.lac.load_keypoint(
            keypoint_pickle_name = "{}__keypoint__.pickle".format(answer_image_name), is_display = False)

        similarity, keypoint_weight = self.lac.similarity_difference(
                                data1 = self.imageDataToken, 
                                data2 = answer_image,
                                should_best_weight_compute = True, 
                                keypoint_weight = None)


        print("keypoint_weight:", keypoint_weight)    
        print("referenced_image_similarity:", similarity)    
        
        return keypoint_weight
    
    def get_best_rank(self, answer_image_name):
        #画像のファイル名とroi_similarity,keypoint_similarityを関連付けて保存
        pickles = self.get_roi_keypoint_similarity()
        #pickles:[[imageData,[sim, deff]],[imageData,[sim, deff]]]
        #これを類似度順にソートする
        #print("merge START")
        #print("similarity_comp end")

        sorted_pickles, keypoint_weight, rank = self.det_rank(pickles, answer_image_name)
        #[{"sile_name", "similarity"}]


        
        
     
        """
        for pickle in sorted_pickles:
            if answer_image_name == pickle["file_name"]:
                print(count, pickle["file_name"], pickle["similarity"])
            count += 1
        """
        print("\tkeypoint_weight:", keypoint_weight)
        #print("rank:",rank)
        print("\t1:", sorted_pickles[0]["file_name"], sorted_pickles[0]["similarity"])
        print("\t{}:{} {}".format(rank, 
                sorted_pickles[rank]["file_name"], sorted_pickles[rank]["similarity"]))
        
    
        return sorted_pickles
    
    def det_rank(self, pickles, answer_image_name):
        #もらった類似度リストから最良の重みを決定する
        best_rank = 99999 #最良の順位        
        keypoint_weight = -1    #最良時の重み
        


        ratio = 0.1         #最初0.1刻み
        weight = 0.5  #0.5から前後に0.1刻み
        sorted_pickles = []
        is_end = False
        while ratio >= 0.01:
            if is_end:
                #print("ddd")
                break
            if  weight >= ratio * 5:    #weight以下0.5の探索を行えるなら 
                weight -= ratio * 5
            for i in range(0, 11):   #10回繰り返す
                if weight > 1.0:    #1超えたら終了
                    #print("a")
                    is_end = True
                    break
                temporary_pickles = []
                #最終的な類似度を入れていく
                for pickle in pickles:
                    temporary_pickle ={
                        "similarity": pickle["keypoint_sim"] * weight + pickle["roi_sim"] * (1 - weight),
                        "file_name" : pickle["file_name"]
                    }
                    temporary_pickles.append(temporary_pickle)

                sorted_temp_pickles = self.merge_sort_dic(temporary_pickles)    #類似度順にソート

                temporary_rank = self.get_reference_image_rank(sorted_temp_pickles, answer_image_name)
                
                #print("tmp_rank:{}, weight{}".format(temporary_rank, weight))
                
                if temporary_rank < best_rank: 
                   best_rank = temporary_rank
                   keypoint_weight = weight
                   sorted_pickles = sorted_temp_pickles


                weight += ratio
            
            ratio /= 10
            weight = keypoint_weight    #最良時の重みに照準を合わせる→前後調査
            
        



        return sorted_pickles, keypoint_weight, best_rank
    
    def get_reference_image_rank(self, sorted_temp_pickles, answer_image_name):
        #線形探索(糞)で参照画像の順位を取得
        rank = 1
        #print("first_rank:", rank)
        #print("len_pickles", len(sorted_temp_pickles))

        for pickle in sorted_temp_pickles:
            #print(pickle["file_name"])
            #print(answer_image_name)
            if pickle["file_name"] == answer_image_name:
                #print("rank:", rank)
                return rank
            rank += 1




        return rank

    
    def merge_sort_dic(self, arr):
        if len(arr) <= 1:
            return arr

        mid = len(arr) // 2
        # ここで分割を行う
        left = arr[:mid]
        right = arr[mid:]

        # 再帰的に分割を行う
        left = self.merge_sort_dic(left)
        right = self.merge_sort_dic(right)

        # returnが返ってきたら、結合を行い、結合したものを次に渡す
        return self.merge_dic(left, right)


    def merge_dic(self, left, right):
        merged = []
        l_i, r_i = 0, 0

        # ソート済み配列をマージするため、それぞれ左から見ていくだけで良い
        while l_i < len(left) and r_i < len(right):

            # ここで=をつけることで安定性を保っている   >=:類似度高い順ソート
            if left[l_i]["similarity"] >= right[r_i]["similarity"]:

                merged.append(left[l_i])
                l_i += 1
            else:
                merged.append(right[r_i])
                r_i += 1

        # 上のwhile文のどちらかがFalseになった場合終了するため、あまりをextendする
        if l_i < len(left):
            merged.extend(left[l_i:])
        if r_i < len(right):
            merged.extend(right[r_i:])
        return merged

     
    def get_roi_keypoint_similarity(self):
        pickles = []                #画像データと類似度の入れ物
       
        os.chdir('./pickle_keypoint')

        file_names = glob.glob("*.pickle")
        os.chdir('../')

        for file_name in file_names:
            
            keypoint_result = self.lac.load_keypoint(keypoint_pickle_name = file_name, 
                                            is_display = False)
            pickle_similarity = {"file_name" : keypoint_result.file_name}
    
            roi_similarity, keypoint_similarity = self.lac.similarity_difference(
                            data1 = self.imageDataToken, 
                            data2 = keypoint_result,
                            should_best_weight_compute = False, 
                            keypoint_weight = None,
                            is_disclete_sim = True)

            pickle_similarity["roi_sim"] = roi_similarity
            pickle_similarity["keypoint_sim"] = keypoint_similarity
                
            pickles.append(pickle_similarity)

            #############################################data1に構図データ、data2にデータベースの画像群#######################################################################################################



        return pickles


"""
def main():
    answer_image_name_list = [
        "000030708.jpg",        #開脚
        "000337885.jpg",        #走ってる二人
        "005142581.jpg",        #BeFIT
        "007989940.jpg",        #肉
        "008922217.jpg",         #握手
        "008449010.jpg"        #スマッシュ
        ] 
    answer_tag = [
        "開脚",
        "走ってる二人",
        "BeFit",
        "肉",
        "握手",
        "スマッシュ",
    ]
    search_image_name = "kouzudata_azuma_02.pickle"
    answer_image_name =  answer_image_name_list[1]
    print("search_image_name:{}".format(search_image_name))
    print("answer_image_name:{}".format(answer_image_name))
    print("start")
    start = time.time()
    search_sort = Search_sort(search_image_name = search_image_name)
    
    keypoint_weight = search_sort.get_best_keypoint_weight(answer_image_name)
    
    return 
 

    search_sort.sort(keypoint_weight = 0.9, answer_image_name = answer_image_name)
    end = time.time()
    print("serch_time:{}".format(end - start))



    return
"""

def main_best_rank():
    #順位が最良の結果を返す
    answer_image_name_list = [
        "000030708.jpg",        #開脚
        "000337885.jpg",        #走ってる二人
        "005142581.jpg",        #BeFIT
        "007989940.jpg",        #肉
        "008922217.jpg",         #握手
        "008449010.jpg"        #スマッシュ
        ] 
    answer_tag = [
        "開脚",
        "走ってる二人",
        "BeFit",
        "肉",
        "握手",
        "スマッシュ",
    ]
    search_image_name = "kouzudata_yoshii_16.pickle"
    answer_image_name =  answer_image_name_list[3]
    print("search_image_name:{}".format(search_image_name))
    print("answer_image_name:{}".format(answer_image_name))
    #print("start")
    start = time.time()
    search_sort = Search_sort(search_image_name = search_image_name)
    
    

    search_sort.get_best_rank(answer_image_name = answer_image_name)
    
    end = time.time()
    print("\tserch_time:{}".format(end - start))





    return



if __name__ == "__main__" :
    main_best_rank()

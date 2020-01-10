import os
import sys
sys.setrecursionlimit(5000)
import tkinter
from tkinter import font
from tkinter import ttk
from tkinter import *
import numpy as np
import copy

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



import loadAndComparison as Lac
from loadAndComparison import ImageData
from loadAndComparison import LoadAndComparison
from loadAndComparison import Keypoint_result

from visualize_keypoint import Visualize_IoU
import glob

from search_sort import Search_sort

import copy

class FontSizer:
    def __init__(self, name='TkDefaultFont', size=12):
        self.name = name
        self.size = size

    def scale(self, sf):
        self.size *= sf

    @property
    def font(self):
        return self.name, int(self.size)

class ROI():
    def __init__(self, value_list, class_name, keypoint, tags, combo_box):
        self.roi = value_list
        self.class_name = class_name
        self.keypoint = keypoint
        self.tags = tags
        self.combo_box = combo_box
        
        return

"""
class Keypoint():
    def __init__(self):


        return

class imageData():
    def __init__(self, roi, class_name = None, keypoint = None):
        self.roi = roi
        self.class_name = class_name
        self.keypoint = keypoint
        self.score = np.full(1, self.roi.shape(0))
        self.sizeX = roi[3] - roi[1]
        self.sizeY = roi[2] - roi[0]

        return


"""

class Size_window(tkinter.Frame):

    def __init__(self, main_window, master=None):
        self.master = master
        self.main_window = main_window


        super().__init__(master)
        self.pack()
        self.create_widgets()
        return

    def create_widgets(self):
        #サイズを入力してください
        self.message = tkinter.Label(self, text='キャンバスの大きさを入力してください')
        self.message.pack(side="top")

        #テキストボックスX
        L1 = tkinter.Label(self, text="X:")
        L1.pack(side=tkinter.LEFT)
        self.E1 = tkinter.Entry(self, bd=1)
        self.E1.pack(side=tkinter.LEFT)
       
        
        #テキストボックスY
        L2 = tkinter.Label(self, text="Y:")
        L2.pack(side=tkinter.LEFT)
        self.E2 = tkinter.Entry(self, bd=1)
        self.E2.pack(side=tkinter.LEFT)
        
        
        #完了ボタン(masterのウィンドウサイズ変更してキャンバスを生成))
        complete = tkinter.Button(self, text="完了",
                         command=self.size_complete)  
                        
        complete.pack(side=tkinter.BOTTOM)
        return

            
    def size_complete(self):
        x = 0
        y = 0
        try:
            x = int(self.E1.get())
            y = int(self.E2.get())
        except ValueError:
            self.E1.delete(0, tkinter.END)
            self.E2.delete(0, tkinter.END)
            self.message["text"] = '有効な値を入力してください'
            self.message["fg"]="red"

            return

        #print(self.E1.get())
        #print(self.E2.get())
        #ウィンドウのサイズを変換
        self.main_window.master.geometry("{}x{}".format(x, y))

        sub = 75
        self.main_window.canvas.config(width = x, height= y )
        #self.main_window.canvas.delete()
        self.main_window.canvas.create_rectangle(0, 0, x, y,
                                                 fill = 'white')
        self.main_window.sizeX = x
        self.main_window.sizeY = y                         
        
        #ウィンドウを破棄
        self.master.destroy()
        return






######################################kokokara"""""""""""""""""""""""""""""""""""""""""

class Application(tkinter.Frame):

    def __init__(self, master, sizeX, sizeY):
        self.class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
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
        self.roi_list = []
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.master = master
        self.mode = self.change_mode("ROI", is_first=True)
        self.selected_roi_number = None                #現在選択されている矩形
        self.selected_corner_number = None
        self.selected_pose_number = None
        self.selected_keypoint_number = None
        self.clicked_editing_keypoint_number = None
        self.clicked_editing_corner_number = None
        self.tag_number = 0
        self.tags = [
                    "tag0_" + str(self.tag_number),    #左上
                    "tag1_" + str(self.tag_number),     #右上
                    "tag2_" + str(self.tag_number),     #右下？
                    "tag3_" + str(self.tag_number),     #左下？
                    "tag4_" + str(self.tag_number),     #矩形
                    "tag5_" + str(self.tag_number),     #クラス名      
                    "tag6_" + str(self.tag_number),     #キーポイント：顔
                    "tag7_" + str(self.tag_number),     #キーポイント：右肩
                    "tag8_" + str(self.tag_number),     #キーポイント：左肩
                    "tag9_" + str(self.tag_number),     #キーポイント：喉
                    "tag10_" + str(self.tag_number),    #キーポイント：右ひじ 
                    "tag11_" + str(self.tag_number),    #キーポイント：左ひじ
                    "tag12_" + str(self.tag_number),    #キーポイント：右手首
                    "tag13_" + str(self.tag_number),    #キーポイント：左手首
                    "tag14_" + str(self.tag_number),    #キーポイント：右尻
                    "tag15_" + str(self.tag_number),    #キーポイント：左尻
                    "tag16_" + str(self.tag_number),    #キーポイント：右ひざ
                    "tag17_" + str(self.tag_number),    #キーポイント：左ひざ
                    "tag18_" + str(self.tag_number),    #キーポイント：右くるぶし
                    "tag19_" + str(self.tag_number),    #キーポイント：左くるぶし
                    "tag20_" + str(self.tag_number),    #ベクトル：顔→喉
                    "tag21_" + str(self.tag_number),    #ベクトル：喉→右肩
                    "tag22_" + str(self.tag_number),    #ベクトル：喉→左肩
                    "tag23_" + str(self.tag_number),    #ベクトル：右肩→右ひじ
                    "tag24_" + str(self.tag_number),    #ベクトル：右ひじ→右手首
                    "tag25_" + str(self.tag_number),    #ベクトル：左肩→左ひじ
                    "tag26_" + str(self.tag_number),    #ベクトル：左ひじ→左手首
                    "tag27_" + str(self.tag_number),    #ベクトル：喉→右尻
                    "tag28_" + str(self.tag_number),    #ベクトル：喉→左尻
                    "tag29_" + str(self.tag_number),    #ベクトル：右尻→左尻
                    "tag30_" + str(self.tag_number),    #ベクトル：右尻→右ひざ
                    "tag31_" + str(self.tag_number),    #ベクトル：右ひざ→右くるぶし
                    "tag32_" + str(self.tag_number),    #ベクトル：左尻→左ひざ
                    "tag33_" + str(self.tag_number),    #ベクトル：左ひざ→左くるぶし
                    "tag34_" + str(self.tag_number),    #選択表示の矩形
                    "tag35_" + str(self.tag_number),    #選択矩形の左上
                    "tag36_" + str(self.tag_number),    #選択矩形の右上
                    "tag37_" + str(self.tag_number),    #選択矩形の左下
                    "tag38_" + str(self.tag_number),    #選択矩形の右下
                    "tag39_" + str(self.tag_number),    #選択矩形の回転矢印
                    "tag40_" + str(self.tag_number),    #選択矩形(各キーポイント) 
                    ]

        self.ini_size = 1
        self.font_sizer = FontSizer(size=12)
        self.is_keypint_editing = False

        #self.after_det_class_name = False
        

        super().__init__(master)
        self.pack()
        self.create_widgets()
        return

    def create_widgets(self):
        

        menubar = tkinter.Menu(self.master)
        
        
        filemenu = tkinter.Menu(menubar, tearoff=0)
        filemenu.add_command(label="New", command=self.define_image_size)
        filemenu.add_command(label="Save", command=self.save_data) 
        filemenu.add_separator()    #メニューのセパレータを追加
        filemenu.add_command(label="Exit", command=self.master.destroy)   
        menubar.add_cascade(label="File", menu=filemenu)
        self.master.config(menu=menubar)


      
        self.det_roi = tkinter.Button(self, text="roi",
                         command=self.define_roi)    #roi決定モード
        self.det_class_name = tkinter.Button(self, text="class",
                         command=self.define_class_name)    #キーポイント決定モード
        self.det_keypoint = tkinter.Button(self, text="person",
                         command=self.change_keypoint)    #キーポイント決定モード
        self.select_roi = tkinter.Button(self, text="select",
                         command=self.change_roi)  

        
        self.det_roi.pack(side="left")
        self.det_class_name.pack(side="left")
        self.det_keypoint.pack(side="left")
        self.select_roi.pack(side="left")
        #キャンバスエリア
        sub = 75
        self.canvas = tkinter.Canvas(self.master,
                             width = self.sizeX, height = self.sizeY)
        self.canvas.create_rectangle(0, 0, self.sizeX, self.sizeY-sub,
                                                 fill = 'white')#塗りつぶし
        #キャンバスバインド
        self.canvas.place(x=0, y=sub)
        #self.canvas.pack()
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        self.canvas.bind("<Button1-Motion>", self.motion)
        
        self.canvas.bind("<Button-3>", self.click_left)
        self.canvas.bind("<ButtonRelease-3>", self.release_left)
        self.canvas.bind("<Button3-Motion>", self.motion_left)
        #self.canvas.bind("<MouseWheel>", self.zoomer)

        self.menu_top = Menu(self.master,tearoff=False)
        #self.menu_2nd = Menu(self.menu_top,tearoff=0)
        #self.menu_3rd = Menu(self.menu_top,tearoff=0)
        #def remove_roi(self):

        self.menu_top.add_cascade (label='削除',command = self.remove_roi,under=5)
        self.menu_top.add_cascade (label='クラス名を指定',command = self.add_class_name,under=5)
        self.menu_top.add_cascade (label='キーポイント',command = self.add_keypoint,under=5)
        
        """
        self.menu_top.add_cascade (label='FILE(F)',menu=self.menu_2nd,under=5)
        self.menu_top.add_separator()
        #self.menu_top.add_command(label='EDIT(E)',underline=5,command=callback)

        self.menu_2nd.add_command(label='New Window(W)',under=4)
        self.menu_2nd.add_cascade(label='Open(O)',under=5,menu=self.menu_3rd)

        self.menu_3rd.add_command(label='Local File(L)',under=11)
        self.menu_3rd.add_command(label='Network(N)',under=8)
        
        """

        self.mode_label = tkinter.Label(self.master, text="mode:{}".format(self.mode))          
        self.mode_label.pack(anchor = NE)
     



        return
    def zoomer(self, event):
        sf = 1.1 if event.delta > 0 else 0.9

        # 元に戻せるように逆数を保持
        
        self.ini_size = self.ini_size / sf
        self.sizeX *= sf
        self.sizeY *= sf

        for roi_obj in self.roi_list:
            roi_obj.roi[0] *= sf
            roi_obj.roi[1] *= sf
            roi_obj.roi[2] *= sf
            roi_obj.roi[3] *= sf
            self.canvas.itemconfigure(roi_obj.tags[5], font=self.font_sizer.font)
        

        self.canvas.scale("all", 0, 0, sf, sf)
        self.font_sizer.scale(sf)
        
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def shoki_size(self):
        self.canvas.scale("all", 0, 0, self.ini_size, self.ini_size)
        self.font_sizer.scale(self.ini_size)
        #self.canvas.itemconfigure('bangou', font=self.font_sizer.font)
        self.canvas.configure(scrollregion=(0,0,500,500)) # 初期スクロール位置。適当に戻す

        self.ini_size = 1


    def clicked_roi(self, event):
        #self.canvas.create_oval(event.x-5, event.y-5,
        #             event.x+5, event.y+5, fill="red",
        #              width=0, tags = self.tags[0])
        #self.preX = event.x
        #self.preY = event.y
        #print(self.roi_tags)
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        self.change_mode("ROI")
        return

    def dragged_roi(self,event):
        #tagtag
        for tag in self.tags:
            self.canvas.delete(tag)

        self.make_roi_instance(self.preX, self.preY, event.x, event.y,
                    self.tags)
         
        
  
    
        #print(self.roi_tags)
        self.master.update()        
        
        return

    def released_roi(self, event):
        #self.canvas.create_oval(event.x-5, event.y-5,
        #             event.x+5, event.y+5, fill="green", width=0)

        #self.postX = event.x
        #self.postY = event.y
        
        if abs(self.preY - self. postY) == 0 and \
            abs(self.preX - self. postX) == 0:
            #print("実行")
            self.canvas.delete(self.tags[0])
            #self.det_selected_roi(self.preX, self.preY, pre_mode = "ROI")
            #self.det_selected_oval(self.preX, self.preY, pre_mode = "ROI")
            
        else:   
            

            if self.preY > self. postY:
                self. postY, self.preY = self.preY, self. postY

            if self.preX > self. postX:
                self. postX, self.preX = self.preX, self. postX

            
            self.make_roi_instance(self.preX, self.preY, self.postX, self.postY,
                        self.tags, is_release = True)
            #print("after make roi_instance",roi_instance.tags)
            
            
            #print(self.roi_tags)

            #print(self.circles_tag)
            #print(self.roi_tags)
            #print("tags of roi_instance after adding 1", roi_instance.tags)

            #print(self.roi_tags)

            
        self.change_mode("SELECT_ROI")
        return
       
    def make_roi_instance(self, preX, preY, postX, postY, tags, is_release = False):
        """
        if tags == None:
            tags = self.tags
        """
        #print("mroi:",tags)
        copied_tags = copy.copy(tags)
        box = None
        upper_left = self.canvas.create_oval(preX-5, preY-5,
                     preX+5, preY+5, fill="red",
                      width=0, tags = copied_tags[0])

        upper_right = self.canvas.create_oval(postX-5, preY-5,
                     postX+5, preY+5, fill="yellow", 
                     width=0, tags = copied_tags[1])
        
        lower_right = self.canvas.create_oval(postX-5, postY-5,
                     postX+5, postY+5, fill="blue", 
                     width=0, tags = copied_tags[2])
        
        lower_left = self.canvas.create_oval(preX-5, postY-5,
                     preX+5, postY+5, fill="green", 
                     width=0, tags = copied_tags[3])
        
        rect = self.canvas.create_rectangle(preX, preY,
                    postX, postY, tags = copied_tags[4],stipple="gray50")#塗りつぶし
        
        self.canvas.delete(copied_tags[5])
        #text = self.canvas.create_text(preX, preY, text = "token",
        #            font = ('FixedSys', 14), tags = copied_tags[5], anchor = NW)

        if is_release:
            #print("ccccccccccccccccccccccc")
            #rect = self.canvas.create_rectangle(preX, preY,
            #        postX, postY, fill = "green", tags = copied_tags[4])#塗りつぶし
            circle = np.array([upper_left, upper_right, lower_left, lower_right])

            roi_array = np.array([preY, preX, postY, postX])
            class_name = None

            roi_instance = ROI(value_list = roi_array, class_name = None, keypoint = None,
                        tags =  copied_tags, combo_box = None)
            self.roi_list.append(roi_instance)          #インスタンスの追加

            self.det_selected_roi(self.postX, self.postY)

            roi_instance.combo_box = self.add_class_name()

            self.tag_number += 1

        else:
            #print("aaaaaaaaaaaaaaaaaaaaaaa")
            pass
           

        #print(self.roi_list)
        for i in range(len(self.tags)):
            self.tags[i] = "tag{}_".format(i) + str(self.tag_number)
     

       
        
        
        #print("rect",rect)
        #print("aftermake,",copied_tags)

        
        

        return 


    def clicked_class_name(self, event):
        #self.canvas.create_oval(event.x-30, event.y-30,
        #             event.x+30, event.y+30, fill="green", width=0)
        count = 0
        #for roi in in self.roi_list:
        pre_roi = self.roi_list[self.selected_roi_number]
        #print(pre_roi.tags[0])
        clicked_roi_number = self.det_selected_roi(event.x, event.y, pre_mode = "CLASS_NAME")

        #print(clicked.tags[0])
        if self.selected_roi_number == clicked_roi_number:
            print("aaaaaaa")
            #クラス名の変更処理
            

        
        return
        
    def released_class_name(self, event):
        
        
        return
    def dragged_class_name(self,event):
        pass

    

    def clicked_keypoint(self, event):
        self.canvas.create_oval(event.x-30, event.y-30,
                     event.x+30, event.y+30, fill="blue", width=0)

        """
        PART_STR = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
                "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
                "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
        鼻
        両目
        両耳
        両肩
        両肘
        両手首
        両尻
        両ひざ
        両くるぶし
        左→右の順番
        """
        
        return
    def released_keypoint(self, event):
        self.canvas.create_oval(event.x-30, event.y-30,
                     event.x+30, event.y+30, fill="red", width=0)
        
        return   
    def dragged_keypoint(self,event):
        
        return
    
    def plain_menu(self, event):
        #何もないところを右クリック
        """
        キャンバスのリサイズ
        保存
        新規

        """
        return

    def roi_menu(self,event):
        """
        矩形or角をクリック
        削除
        コピー
        貼り付け
        キーポイント新規 
        クラス選択   

        キーポイントに接していたら
            キーポイント削除

        """
        self.show_popup(event)


        
        return
    def show_popup(self, event):
        self.menu_top.post(self.preX, self.preY)    #変な場所にメニューが出る
        return
    def remove_roi(self):
        #print("uwa-")
        #選択した矩形の削除
        if self.selected_roi_number == None:
            print("選択されていません")
            return
        tags = self.roi_list[self.selected_roi_number].tags
        for tag in tags:
             self.canvas.delete(tag)
            
        self.roi_list[self.selected_roi_number].\
                        combo_box.place(x = -1000,y = -1000)

        self.roi_list.pop(self.selected_roi_number)
        """
        for roi_obj in self.roi_list:
            print("roi")
            print(roi_obj.roi)
            print("class")
            print(roi_obj.class_name)
            print("keypoint")
            print(roi_obj.keypoint)
        """
        #print("self.selected_pose_number", self.selected_pose_number)
        #print("self.selected_roi_number", self.selected_roi_number)
        self.selected_roi_number = None
        self.selected_pose_number = None
        
            


        """
        for roi_obj in self.roi_list:
            print("list after removed", roi_obj.roi)
        """
        return

    def keypoint_center_list(self, center_roi_number):
        """
        self.roi = value_list
        self.class_name = class_name
        self.keypoint = keypoint
        self.tags = tags
        self.combo_box = combo_box
        """
        x_len = self.roi_list[center_roi_number].roi[3] - self.roi_list[center_roi_number].roi[1]
        y_len = self.roi_list[center_roi_number].roi[2] - self.roi_list[center_roi_number].roi[0]

        keypoint_relativePoint_list = [
            [
            int(self.roi_list[center_roi_number].roi[1] + x_len * 0.5),
            int(self.roi_list[center_roi_number].roi[0] + y_len * 0.1)
            ],                                    #顔:x:中心、y：上から2割
            [
            int(self.roi_list[center_roi_number].roi[1] + x_len * 0.75),
            int(self.roi_list[center_roi_number].roi[0] + y_len * 0.3)
            ],                                    #右肩：ｘ：3/4、ｙ：上から3割
            [
            int(self.roi_list[center_roi_number].roi[1] + x_len * 0.25),
            int(self.roi_list[center_roi_number].roi[0] + y_len * 0.3)
            ],                                       #左肩：ｘ：左から1/4、ｙ：上から3割
            [
            int(self.roi_list[center_roi_number].roi[1] + x_len * 0.5),
            int(self.roi_list[center_roi_number].roi[0] + y_len * 0.3)
            ],                                      #喉：ｘ：中心、ｙ：上から3割
            [
            int(self.roi_list[center_roi_number].roi[1] + x_len * 0.75),
            int(self.roi_list[center_roi_number].roi[0] + y_len * 0.45)
            ],                                      #右ひじ：ｘ：3/4、ｙ：上から４.5割
            [
            int(self.roi_list[center_roi_number].roi[1] + x_len * 0.25),
            int(self.roi_list[center_roi_number].roi[0] + y_len * 0.45)
            ],                                      #左ひじ：ｘ：左から1/4、ｙ：上から４.5割
            [
            int(self.roi_list[center_roi_number].roi[1] + x_len * 0.75),
            int(self.roi_list[center_roi_number].roi[0] + y_len * 0.6)
            ],                                       #右手首：ｘ：3/4、ｙ：上から6割
            [
            int(self.roi_list[center_roi_number].roi[1] + x_len * 0.25),
            int(self.roi_list[center_roi_number].roi[0] + y_len * 0.6)
            ],                                       #左手首：ｘ：左から1/4、ｙ：上から6割
            [
            int(self.roi_list[center_roi_number].roi[1] + x_len * 0.6),
            int(self.roi_list[center_roi_number].roi[0] + y_len * 0.6)
            ],                                          #右尻：ｘ：3/５、ｙ：上から6割
            [
            int(self.roi_list[center_roi_number].roi[1] + x_len * 0.4),
            int(self.roi_list[center_roi_number].roi[0] + y_len * 0.6)
            ],                                          #左尻：ｘ：２/５、ｙ：上から6割
            [
            int(self.roi_list[center_roi_number].roi[1] + x_len * 0.6),
            int(self.roi_list[center_roi_number].roi[0] + y_len * 0.8)
            ],                                           #右ひざ：ｘ：3/５、ｙ：上から8割
            [
            int(self.roi_list[center_roi_number].roi[1] + x_len * 0.4),
            int(self.roi_list[center_roi_number].roi[0] + y_len * 0.8)
            ],                                           #左ひざ：ｘ：/５、ｙ：上から8割
            [
            int(self.roi_list[center_roi_number].roi[1] + x_len * 0.6),
            int(self.roi_list[center_roi_number].roi[0] + y_len * 0.99)
            ],                                          #右くるぶし：ｘ：3/５、ｙ：上から９．９割
            [
            int(self.roi_list[center_roi_number].roi[1] + x_len * 0.4),
            int(self.roi_list[center_roi_number].roi[0] + y_len * 0.99)
            ],                                           #左くるぶし：ｘ：２/５、ｙ：上から９．９割
        ]

        return  keypoint_relativePoint_list
        
    def kyeypoint_vector_list(self, keypoint_relativePoint_list):
 
        vector_list = [
            np.array([keypoint_relativePoint_list[0][1],keypoint_relativePoint_list[0][0],
            keypoint_relativePoint_list[3][1],keypoint_relativePoint_list[3][0]]), #ベクトル：顔→喉

            np.array([keypoint_relativePoint_list[3][1],keypoint_relativePoint_list[3][0],
            keypoint_relativePoint_list[1][1],keypoint_relativePoint_list[1][0]]),#ベクトル：喉→右肩
            
            np.array([keypoint_relativePoint_list[3][1],keypoint_relativePoint_list[3][0],
            keypoint_relativePoint_list[2][1],keypoint_relativePoint_list[2][0]]),#ベクトル：喉→左肩          
 
            np.array([keypoint_relativePoint_list[1][1],keypoint_relativePoint_list[1][0],
            keypoint_relativePoint_list[4][1],keypoint_relativePoint_list[4][0]]),#ベクトル：右肩→右ひじ         

            np.array([keypoint_relativePoint_list[4][1],keypoint_relativePoint_list[4][0],
            keypoint_relativePoint_list[6][1],keypoint_relativePoint_list[6][0]]),#ベクトル：右ひじ→右手首         

            np.array([keypoint_relativePoint_list[2][1],keypoint_relativePoint_list[2][0],
            keypoint_relativePoint_list[5][1],keypoint_relativePoint_list[5][0]]),#ベクトル：左肩→左ひじ         

            np.array([keypoint_relativePoint_list[5][1],keypoint_relativePoint_list[5][0],
            keypoint_relativePoint_list[7][1],keypoint_relativePoint_list[7][0]]),#ベクトル：左ひじ→左手首         

            np.array([keypoint_relativePoint_list[3][1],keypoint_relativePoint_list[3][0],
            keypoint_relativePoint_list[8][1],keypoint_relativePoint_list[8][0]]),#ベクトル：喉→右尻   
            
            np.array([keypoint_relativePoint_list[3][1],keypoint_relativePoint_list[3][0],
            keypoint_relativePoint_list[9][1],keypoint_relativePoint_list[9][0]]),#ベクトル：喉→左尻   

            np.array([keypoint_relativePoint_list[9][1],keypoint_relativePoint_list[8][0],
            keypoint_relativePoint_list[9][1],keypoint_relativePoint_list[9][0]]),#ベクトル：右尻→左尻 

            np.array([keypoint_relativePoint_list[8][1],keypoint_relativePoint_list[8][0],
            keypoint_relativePoint_list[10][1],keypoint_relativePoint_list[10][0]]),#ベクトル：右尻→右ひざ

            np.array([keypoint_relativePoint_list[10][1],keypoint_relativePoint_list[10][0],
            keypoint_relativePoint_list[12][1],keypoint_relativePoint_list[12][0]]),#ベクトル：右ひざ→右くるぶし
            
            np.array([keypoint_relativePoint_list[9][1],keypoint_relativePoint_list[9][0],
            keypoint_relativePoint_list[11][1],keypoint_relativePoint_list[11][0]]),#ベクトル：左尻→左ひざ

            np.array([keypoint_relativePoint_list[11][1],keypoint_relativePoint_list[11][0],
            keypoint_relativePoint_list[13][1],keypoint_relativePoint_list[13][0]]),#ベクトル：左ひざ→左くるぶし




        ]

        return  vector_list
    def vector_editing_roi(self):
        #
        return

    def get_pose_editing_roi_corner(self):
        #編集矩形の四隅を決定
        keypoints = self.roi_list[self.selected_pose_number].keypoint
        roi = self.roi_list[self.selected_pose_number].roi
        min_x = self.sizeX
        min_y = self.sizeY
        max_x = 0
        max_y = 0
        #編集矩形の範囲の設定
        x_len = roi[3] - roi[1]
        y_len = roi[2] - roi[0]

        count = 0
        for keypoint in keypoints:
            if count == 0:
                radius = x_len * (10 / 100)
            else:
                radius = x_len * (5 / 100)

            if keypoint[0] - radius < min_x:
                min_x = keypoint[0] - radius

            if keypoint[1] - radius < min_y:
                min_y = keypoint[1] - radius

            if max_x < keypoint[0] + radius:
                max_x = keypoint[0] + radius
            
            if max_y < keypoint[1] + radius:
                max_y = keypoint[1] + radius
            
            count += 1



        return min_x, min_y, max_x, max_y
    
    def pose_editing_roi(self):
        #選択された姿勢の編集矩形
        #print("self.selected_roi_number",self.selected_roi_number)
        #print("self.select_roi[self.selected_roi_number]",self.roi_list[self.selected_roi_number])
        #print("self.select_roi[self.selected_roi_number].tags",self.roi_list[self.selected_roi_number].tags)

        
        tag = self.roi_list[self.selected_pose_number].tags    #３４番目が編集矩形

        
        """
                            "tag34_" + str(self.tag_number),    #選択表示の矩形
                    "tag35_" + str(self.tag_number),    #選択矩形の左上
                    "tag36_" + str(self.tag_number),    #選択矩形の右上
                    "tag37_" + str(self.tag_number),    #選択矩形の左下
                    "tag38_" + str(self.tag_number),    #選択矩形の右下
                    "tag39_" + str(self.tag_number),    #選択矩形の回転矢印
        """
        min_x, min_y, max_x, max_y = self.get_pose_editing_roi_corner()
        #矩形
        print("min_x, min_y, max_x, max_y")
        print(min_x, min_y, max_x, max_y)
        self.canvas.create_rectangle(min_x, min_y, max_x, max_y,
                                outline = "gray50", tags = tag[34])
        
        #矩形周りの点
        self.canvas.create_oval(min_x-5, min_y-5, min_x+5, min_y+5,         #左上
                                fill = "gray50", tags = tag[35])
        
        #self.canvas.create_oval((min_x+max_x)/2-5, min_y-5,
        #                        (min_x+max_x)/2+5, min_y+5,         #上中段
        #                        fill = "gray50", tags = tag) 

        self.canvas.create_oval(max_x-5, min_y-5, max_x+5, min_y+5,         #右上
                                fill = "gray50", tags = tag[36])

        #self.canvas.create_oval(min_x-5, (min_y+max_y)/2-5,
        #                        min_x+5, (min_y+max_y)/2+5,         #中段左
        #                        fill = "gray50", tags = tag) 
        
        #self.canvas.create_oval(max_x-5, (min_y+max_y)/2-5,
        #                        max_x+5, (min_y+max_y)/2+5,         #中段右
        #                        fill = "gray50", tags = tag) 
        
        self.canvas.create_oval(min_x-5, max_y-5, min_x+5, max_y+5,         #左下
                                fill = "gray50", tags = tag[37]) 
        
        #self.canvas.create_oval((min_x+max_x)/2 -5, max_y-5,
        #                        (min_x+max_x)/2 +5, max_y+5,         #下中段
        #                       fill = "gray50", tags = tag) 
        
        self.canvas.create_oval(max_x-5, max_y-5, max_x+5, max_y+5,         #右下
                                fill = "gray50", tags = tag[38])
        

        #回転矢印の線のほう
        #self.canvas.create_oval((min_x+max_x)/2-10, min_y-40,
        #                        (min_x+max_x)/2+10, min_y-20,         
        #                    outline = "gray50", width = 5, tags = tag[39])
        
        #回転矢印の矢のほう
        #self.canvas.create_polygon((min_x+max_x)/2 + 10, min_y -25,
        #                        (min_x+max_x)/2 , min_y - 35,
        #                        (min_x+max_x)/2 + 20, min_y - 35,
        #                        fill = "gray50", tags = tag[39])
        
        #回転矢印→上中段の点の点線
        #self.canvas.create_line((min_x+max_x)/2, min_y - 30,
        #                        (min_x+max_x)/2, min_y,
        #                        fill = "gray50", tags = tag[39])


        self.is_keypint_editing = False
        return
    
    """
    def create_triangle(self, x1, y1, x2, y2, x3, y3, 
                            tags, fill=None, outline=None, stipple = None):
        
        #三角形の描画
        if fill == None:
            fill = "black"
            stipple = "gray0"
        if outline = None:
            outline = "black"
            stipple = "gray0"
        
        self.canvas.create_line(x1, y1, x2, y2,)

        self.canvas.create_polygon(max_x-20, max_y-20, max_x+20, max_y+20,         #右下
                                outline = "gray50", tags = tag)


        
        
        
        return
    """
    def keypoint_editing_roi(self):
        #選択されたキーポイントの編集矩形
        tag = self.roi_list[self.selected_pose_number].tags    #40番目が編集矩形


        keypoint = self.roi_list[self.selected_pose_number].keypoint[self.selected_keypoint_number]
        roi = self.roi_list[self.selected_pose_number].roi
        min_x = self.sizeX
        min_y = self.sizeY
        max_x = 0
        max_y = 0
        #編集矩形の範囲の設定
        x_len = roi[3] - roi[1]

        if self.selected_keypoint_number == 0:
            radius = x_len * (10 / 100)
        else:
            radius = x_len * (5 / 100)

        min_x = keypoint[0] - radius
        min_y = keypoint[1] - radius
        max_x = keypoint[0] + radius        
        max_y = keypoint[1] + radius
        
        #矩形
        print("min_x, min_y, max_x, max_y")
        print(min_x, min_y, max_x, max_y)
        self.canvas.create_rectangle(min_x, min_y, max_x, max_y,
                                outline = "gray50", tags = tag[40])
        
        #矩形周りの点
        self.canvas.create_oval(min_x-5, min_y-5, min_x+5, min_y+5,         #左上
                                fill = "gray50", tags = tag[40])


        self.canvas.create_oval(max_x-5, min_y-5, max_x+5, min_y+5,         #右上
                                fill = "gray50", tags = tag[40])

        
        self.canvas.create_oval(min_x-5, max_y-5, min_x+5, max_y+5,         #左下
                                fill = "gray50", tags = tag[40]) 

        
        self.canvas.create_oval(max_x-5, max_y-5, max_x+5, max_y+5,         #右下
                                fill = "gray50", tags = tag[40])
        




        self.is_keypint_editing = True

        return

    def add_keypoint(self):             #選択した矩形にキーポイントを描画
        point_list = self.keypoint_center_list(self.selected_roi_number)
        vector_list =  self.kyeypoint_vector_list(point_list)
        tags = self.roi_list[self.selected_roi_number].tags
        roi = self.roi_list[self.selected_roi_number].roi
        pose = []
        count = 0
        x_len = roi[3] - roi[1]
        y_len = roi[2] - roi[0]


        for point in point_list:
            if count == 0:
                radius = x_len * (10 / 100)
            else:
                radius = x_len * (5 / 100)

            self.canvas.create_oval(point[0]-radius, point[1]-radius,
                        point[0]+radius, point[1]+radius, fill="red",
                        width=0, tags = tags[count+6])
            #pose.append(np.array([pose[0], pose[1], True]))
            
            count += 1

        self.roi_list[self.selected_roi_number].keypoint = pose 
                    #決定したキーポイントを格納(そのままでは使えない)
    
        for vector in vector_list:
            self.canvas.create_line(vector[1], vector[0], vector[3], vector[2],
                             width = 2.0, fill = 'blue',tags = tags[count + 6])
            count += 1

        self.roi_list[self.selected_roi_number].keypoint = point_list


        return

    def add_class_name(self):
        #選択した矩形にクラス名を追加
        if self.selected_roi_number == None:
            print("選択されていません")
            return
        
        roi_instance = self.roi_list[self.selected_roi_number]
        if roi_instance.combo_box == None:  #コンボボックスの中身がNoneのとき
            print("add_classname:first combobox")
            val = tkinter.StringVar()




            box = ttk.Combobox(self.master, values = self.class_names, 
                                textvariable=val, state='normal')   
            box.bind('<<ComboboxSelected>>', self.select_class_name_from_box )
            #box.bind("<key>", self.key_input)
           
            box.current(0) #初期値を'正会員(index=0)'に設定   
        else:
            print("add_classname:existing combobox")
            box =  roi_instance.combo_box   #基からあったら使いまわし
            
        
        box.place(x = roi_instance.roi[1], y = roi_instance.roi[0]) #座標指定で表示 
        

        #self.after_det_class_name = True
                
        
        #self.roi_list[self.selected_roi_number].class_name = 
        """
        for roi_obj in self.roi_list:
            print("list after removed", roi_obj.roi)
        """
        #box.place(x = -100, y = -100)
        
        return box
    def key_input(self, event):
        print("ddd")

        return
    def select_class_name_from_box(self, event):
        print("selected...")
        roi_instance = self.roi_list[self.selected_roi_number]

        roi_instance.class_name = roi_instance.combo_box.get() 
        
        print("roi_instance.class_name",roi_instance.class_name)
            #コンボボックス内の値をclass_nameに入れる

          #クラス名を表示
        #rect = self.canvas.create_rectangle(roi[1] + delta_x, roi[0] + delta_y,
        #        roi[3], roi[2],fill = "green", tags = tags[5])#塗りつぶし
        self.canvas.delete(roi_instance.tags[5])
        text = self.canvas.create_text(roi_instance.roi[1], roi_instance.roi[0],
                                        text = roi_instance.class_name,
                                        font = ('FixedSys', 14),
                                        tags = roi_instance.tags[5],
                                        anchor = NW)
        
        self.canvas.itemconfigure(roi_instance.tags[5], text = roi_instance.class_name)

        return
    
    def get_class_name_from_box(self): #クラス名決定後の後処理    
        """
        コンボボックスの排除
        クラス決定直後フラグを下す
        """
        #self.after_det_class_name = False
        
        roi_instance.combo_box.place(x = -100, y = -100)

        return



    def clicked_select_roi(self,event): #矩形がクリックされたときに呼び出される処理
        #矩形を選択
        """
        #選択済みでない矩形をクリック
            #そのままドラッグ　→　矩形の位置変更
            #そのままリリーズ　→　矩形の選択

        #選択済みでない矩形の四隅をクリック　→　矩形を選択            
            #そのままドラッグ　→　矩形のサイズ変更
            #そのままリリーズ　→　矩形の選択
 
                 

        #選択済みの矩形の四隅をクリック
            #そのままドラッグ　→　矩形のサイズ変更
            #リリース　→　クラス選択

        #選択済みの矩形をクリック
            #ドラッグ　→　矩形の位置変更
            #リリース(ダブルクリック)　→　クラス選択
         
        #選択済みでない位置をクリック　→　選択の解除
            #そのままドラッグ　→　何もしない
            #そのままリリーズ　→　何もしない 
        

        #→クリックの時点ではただの選択処理のみ
        """
        #is_rect, is_oval　=  self.det_selected_oval(event.x, event.y, pre_mode = "ROI")
        self.selected_pose_number, self.selected_roi_number, self.selected_corner_number = \
                                self.det_selected_oval(event.x, event.y)
        #print("corner_number",self.selected_corner_number)
        
        return
    
  
    

    def dragged_select_roi(self,event):
        """
        ドラッグパターン
        矩形：位置変更
        四隅：サイズ変更
        何もなし：何もしない
        """
        try:
            tags =  self.roi_list[self.selected_pose_number].tags
            for i in range(6,len(tags)):
                self.canvas.delete(tags[i])
            
            if not self.selected_pose_number == None:
                if not self.clicked_editing_corner_number == None:
                    #編集矩形の四隅がクリックされていれば
                    self.resize_editing_roi(event.x, event.y)      #編集矩形の拡大縮小
                    
                elif not self.clicked_editing_keypoint_number == None and self.is_keypint_editing:
                    #編集矩形内のキーポイントがクリックされていれば
                    self.move_one_point(event.x, event.y)            #キーポイントの位置変更
                    
                else:
                    #編集矩形自体がクリックされていれば
                    self.move_editing_roi(event.x, event.y)         #編集矩形の位置変更
                    
        except:
        

            

            try:
                tags =  self.roi_list[self.selected_roi_number].tags
                for tag in tags:
                    self.canvas.delete(tag)
            except:
                #print("exception")
                return      #どの矩形にも触れていなければ何もしない
            
            if self.selected_corner_number == None: #コーナーをクリックしていない→移動

                self.move_roi_instance(event.x, event.y)
            
            else:                                   #コーナーをクリック→リサイズ
                self.resize_roi_instance(event.x, event.y)

        self.preX = event.x
        self.preY = event.y
       
        self.master.update()      
            


        
        return

    def move_one_point(self, x, y):
        #キーポイントの移動
        #print("\nmove_one_point")
        tags = self.roi_list[self.selected_pose_number].tags
        pose = self.roi_list[self.selected_pose_number].keypoint
        roi = self.roi_list[self.selected_pose_number].roi
        #print("keypoint_number", self.selected_keypoint_number)
        #print("pose", pose)
        #print("pose.keypoint", pose[self.selected_keypoint_number])
        keypoint = pose[self.selected_keypoint_number]
        #print("keypoint",keypoint)

        #print("mroi:",tags)
        #copied_tags = copy.copy(tags)
        delta_x = x - self.preX
        delta_y = y - self.preY


        print("delta:",delta_x, delta_y)
        #キーポイントの値更新
        keypoint[0] += delta_x
        keypoint[1] += delta_y
        i = self.selected_keypoint_number + 6       #タグ番号
        self.canvas.create_oval(keypoint[0]-5, keypoint[1]-5,
                     keypoint[0]+5, keypoint[1]+5, fill="red",
                      width=0, tags = tags[i])


        count = 0
        x_len = roi[3] - roi[1]
        y_len = roi[2] - roi[0]
        for point in pose:
            if count == 0:
                radius = x_len * (10 / 100)
            else:
                radius = x_len * (5 / 100)

            self.canvas.create_oval(point[0]-radius, point[1]-radius,
                        point[0]+radius, point[1]+radius, fill="red",
                        width=0, tags = tags[count+6])  
            count += 1

        vector_list =  self.kyeypoint_vector_list(pose)
        for vector in vector_list:
            self.canvas.create_line(vector[1], vector[0], vector[3], vector[2],
                             width = 2.0, fill = 'blue',tags = tags[count + 6])
            count += 1

        #self.roi_list[self


        
        #if not self.roi_list[self.selected_pose_number].keypoint == None:
        #    self.move_keypoints(delta_x, delta_y)


        return

    def move_editing_roi(self, x, y):
        #編集矩形の移動
        #print("move")
        """
        if tags == None:
            tags = self.tags
        """
        x0, y0, x1, y1 = self.get_pose_editing_roi_corner()
        roi = np.array([y0, x0, y1, x1])
        tags = self.roi_list[self.selected_pose_number].tags
        #print("mroi:",tags)
        #copied_tags = copy.copy(tags)
        delta_x = x - self.preX
        delta_y = y - self.preY
        """
        print("preX",self.preX)
        print("preY",self.preY)
        print("x",x)
        print("y",y)
        print("roi",roi)
        print("xahyouX",roi[1] + delta_x)
        print("xahyouY",roi[0] + delta_y)
        print("")
        """

        rect = self.canvas.create_rectangle(roi[1] + delta_x, roi[0] + delta_y,
                roi[3] + delta_x, roi[2] + delta_y,outline = "gray50", tags = tags[34])
        

        upper_left = self.canvas.create_oval(roi[1]+delta_x-5, roi[0]+delta_y-5,
                     roi[1]+delta_x+5, roi[0]+delta_y+5, fill="gray50",
                      width=0, tags = tags[35])

        
        upper_right = self.canvas.create_oval(roi[3]+delta_x-5, roi[0]+delta_y-5,
                     roi[3]+delta_x+5, roi[0]+delta_y+5, fill="gray50", 
                     width=0, tags = tags[36])

   
        lower_left = self.canvas.create_oval(roi[1]+delta_x-5, roi[2]+delta_y-5,
                     roi[1]+delta_x+5, roi[2]+delta_y+5, fill="gray50", 
                     width=0, tags = tags[37])
        
        lower_right = self.canvas.create_oval(roi[3]+delta_x-5, roi[2]+delta_y-5,
                     roi[3]+delta_x+5, roi[2]+delta_y+5, fill="gray50", 
                     width=0, tags = tags[38])



        changed_roi = np.array([roi[0]+delta_y, roi[1]+delta_x ,roi[2]+delta_y ,roi[3]+delta_x])

        #self.roi_list[self.selected_roi_number].roi = changed_roi

        if not self.roi_list[self.selected_pose_number].keypoint == None:
            #print("ddd")
            self.move_keypoints(delta_x, delta_y)




        return 

    def resize_editing_roi(self, x, y):

        x0, y0, x1, y1 = self.get_pose_editing_roi_corner()
        roi = np.array([y0, x0, y1, x1])
        
        tags = self.roi_list[self.selected_pose_number].tags
        pre_roi = copy.copy(roi)
        #print("mroi:",tags)
        #copied_tags = copy.copy(tags)
        clicked_number = self.selected_pose_number
        #print("click_corner",clicked_number)
        delta_x = x - self.preX
        delta_y = y - self.preY
        array = []
        clicked_number = self.clicked_editing_corner_number
        
        if clicked_number == 0:     #左上
            #print(0)

            rect = self.canvas.create_rectangle(roi[1] + delta_x, roi[0] + delta_y,
                roi[3], roi[2],outline = "gray50", tags = tags[34])#塗りつぶし
            

            upper_left = self.canvas.create_oval(roi[1]+delta_x-5, roi[0]+delta_y-5,
                        roi[1]+delta_x+5, roi[0]+delta_y+5, fill="gray50",
                        width=0, tags = tags[35])

            
            upper_right = self.canvas.create_oval(roi[3]-5, roi[0]+delta_y-5,
                        roi[3]+5, roi[0]+delta_y+5, fill="gray50", 
                        width=0, tags = tags[36])
            
            lower_right = self.canvas.create_oval(roi[3]-5, roi[2]-5,
                     roi[3]+5, roi[2]+5, fill="gray50", 
                     width=0, tags = tags[37])
    
            lower_left = self.canvas.create_oval(roi[1]+delta_x-5, roi[2]-5,
                        roi[1]+delta_x+5, roi[2]+5, fill="gray50", 
                        width=0, tags = tags[38])
            
            array = [roi[0] + delta_y, roi[1] + delta_x, roi[2], roi[3]]

        elif clicked_number == 1:   #右上
            rect = self.canvas.create_rectangle(roi[1], roi[0] + delta_y,
                roi[3] + delta_x, roi[2],outline = "gray50", tags = tags[34])#塗りつぶし
        

            upper_left = self.canvas.create_oval(roi[1]-5, roi[0]+delta_y-5,
                        roi[1]+5, roi[0]+delta_y+5, fill="gray50",
                        width=0, tags = tags[35])

            
            upper_right = self.canvas.create_oval(roi[3]+delta_x-5, roi[0]+delta_y-5,
                        roi[3]+delta_x+5, roi[0]+delta_y+5, fill="gray50", 
                        width=0, tags = tags[36])
            
   
            lower_left = self.canvas.create_oval(roi[1]-5, roi[2]-5, roi[1]+5, roi[2]+5, 
                        fill="gray50", width=0, tags = tags[37])
        

            lower_right = self.canvas.create_oval(roi[3]+delta_x-5, roi[2]-5,
                        roi[3]+delta_x+5, roi[2]+5, fill="gray50", 
                        width=0, tags = tags[38])

            
            array = [roi[0] + delta_y, roi[1], roi[2], roi[3] + delta_x]

        elif clicked_number == 2:   #右下
            rect = self.canvas.create_rectangle(roi[1], roi[0],
                roi[3] + delta_x, roi[2] + delta_y,outline = "gray50", tags = tags[34])#塗りつぶし
        
            upper_left = self.canvas.create_oval(roi[1]-5, roi[0]-5,
                     roi[1]+5, roi[0]+5, fill="gray50",
                      width=0, tags = tags[35])

            upper_right = self.canvas.create_oval(roi[3]+delta_x-5, roi[0]-5,
                        roi[3]+delta_x+5, roi[0]+5, fill="gray50", 
                        width=0, tags = tags[36])
            
            lower_left = self.canvas.create_oval(roi[1]-5, roi[2]+delta_y-5,
                        roi[1]+5, roi[2]+delta_y+5, fill="gray50", 
                        width=0, tags = tags[37])
            
            lower_right = self.canvas.create_oval(roi[3]+delta_x-5, roi[2]+delta_y-5,
                        roi[3]+delta_x+5, roi[2]+delta_y+5, fill="gray50", 
                        width=0, tags = tags[38])
            
            array = [roi[0], roi[1], roi[2] + delta_y, roi[3] + delta_x]

        elif clicked_number == 3:   #左下
            rect = self.canvas.create_rectangle(roi[1] + delta_x, roi[0],
                roi[3], roi[2] + delta_y,outline = "gray50", tags = tags[34])#塗りつぶし
        

            upper_left = self.canvas.create_oval(roi[1]+delta_x-5, roi[0]-5,
                        roi[1]+delta_x+5, roi[0]+5, fill="gray50",
                        width=0, tags = tags[35])
            
            upper_right = self.canvas.create_oval(roi[3]-5, roi[0]-5,
                     roi[3]+5, roi[0]+5, fill="gray50", 
                     width=0, tags = tags[36])

    
            lower_left = self.canvas.create_oval(roi[1]+delta_x-5, roi[2]+delta_y-5,
                        roi[1]+delta_x+5, roi[2]+delta_y+5, fill="gray50", 
                        width=0, tags = tags[37])
            
            lower_right = self.canvas.create_oval(roi[3]-5, roi[2]+delta_y-5,
                        roi[3]+5, roi[2]+delta_y+5, fill="gray50", 
                        width=0, tags = tags[38])

            array = [roi[0], roi[1] + delta_x, roi[2]+delta_y, roi[3]]
        else:
            print("sine")

        changed_roi = np.array(array)

        #ひっくり返ったときの処理
        #→releaseの時に直す

        #self.roi_list[self.selected_roi_number].roi = changed_roi

        if not self.roi_list[self.selected_pose_number].keypoint == None:
            self.resize_keypoints(pre_roi = pre_roi, post_roi = changed_roi)

        return

    def resize_roi_instance(self, x, y):        #
        

        
        roi = self.roi_list[self.selected_roi_number].roi
        tags = self.roi_list[self.selected_roi_number].tags
        pre_roi = copy.copy(roi)
        #print("mroi:",tags)
        #copied_tags = copy.copy(tags)
        clicked_number = self.selected_corner_number
        #print("click_corner",clicked_number)
        delta_x = x - self.preX
        delta_y = y - self.preY
        array = []
        
        
        if clicked_number == 0:     #左上
            #print(0)

            rect = self.canvas.create_rectangle(roi[1] + delta_x, roi[0] + delta_y,
                roi[3], roi[2],fill = "green", tags = tags[4],stipple="gray50")#塗りつぶし
            

            upper_left = self.canvas.create_oval(roi[1]+delta_x-5, roi[0]+delta_y-5,
                        roi[1]+delta_x+5, roi[0]+delta_y+5, fill="red",
                        width=0, tags = tags[0])

            
            upper_right = self.canvas.create_oval(roi[3]-5, roi[0]+delta_y-5,
                        roi[3]+5, roi[0]+delta_y+5, fill="yellow", 
                        width=0, tags = tags[1])
            
            lower_right = self.canvas.create_oval(roi[3]-5, roi[2]-5,
                     roi[3]+5, roi[2]+5, fill="green", 
                     width=0, tags = tags[2])
    
            lower_left = self.canvas.create_oval(roi[1]+delta_x-5, roi[2]-5,
                        roi[1]+delta_x+5, roi[2]+5, fill="blue", 
                        width=0, tags = tags[3])
            
            text = self.canvas.create_text(roi[1]+delta_x, roi[0]+delta_y,
                        text = self.roi_list[self.selected_roi_number].class_name,
                        font = ('FixedSys', 14),
                        tags = tags[5],
                        anchor = NW)
            
            array = [roi[0] + delta_y, roi[1] + delta_x, roi[2], roi[3]]

        elif clicked_number == 1:   #右上
            rect = self.canvas.create_rectangle(roi[1], roi[0] + delta_y,
                roi[3] + delta_x, roi[2],fill = "green", tags = tags[4],stipple="gray50")#塗りつぶし
        

            upper_left = self.canvas.create_oval(roi[1]-5, roi[0]+delta_y-5,
                        roi[1]+5, roi[0]+delta_y+5, fill="red",
                        width=0, tags = tags[0])

            
            upper_right = self.canvas.create_oval(roi[3]+delta_x-5, roi[0]+delta_y-5,
                        roi[3]+delta_x+5, roi[0]+delta_y+5, fill="yellow", 
                        width=0, tags = tags[1])
            
   
            lower_left = self.canvas.create_oval(roi[1]-5, roi[2]-5, roi[1]+5, roi[2]+5, 
                        fill="blue", width=0, tags = tags[3])
        

            lower_right = self.canvas.create_oval(roi[3]+delta_x-5, roi[2]-5,
                        roi[3]+delta_x+5, roi[2]+5, fill="green", 
                        width=0, tags = tags[2])

            text = self.canvas.create_text(roi[1], roi[0]+delta_y,
                        text = self.roi_list[self.selected_roi_number].class_name,
                        font = ('FixedSys', 14),
                        tags = tags[5],
                        anchor = NW)
            
            array = [roi[0] + delta_y, roi[1], roi[2], roi[3] + delta_x]

        elif clicked_number == 2:   #右下
            rect = self.canvas.create_rectangle(roi[1], roi[0],
                roi[3] + delta_x, roi[2] + delta_y,fill = "green", tags = tags[4],stipple="gray50")#塗りつぶし
        
            upper_left = self.canvas.create_oval(roi[1]-5, roi[0]-5,
                     roi[1]+5, roi[0]+5, fill="red",
                      width=0, tags = tags[0])

            upper_right = self.canvas.create_oval(roi[3]+delta_x-5, roi[0]-5,
                        roi[3]+delta_x+5, roi[0]+5, fill="yellow", 
                        width=0, tags = tags[1])
            
            lower_left = self.canvas.create_oval(roi[1]-5, roi[2]+delta_y-5,
                        roi[1]+5, roi[2]+delta_y+5, fill="blue", 
                        width=0, tags = tags[3])
            
            lower_right = self.canvas.create_oval(roi[3]+delta_x-5, roi[2]+delta_y-5,
                        roi[3]+delta_x+5, roi[2]+delta_y+5, fill="green", 
                        width=0, tags = tags[2])

            text = self.canvas.create_text(roi[1], roi[0],
                        text = self.roi_list[self.selected_roi_number].class_name,
                        font = ('FixedSys', 14),
                        tags = tags[5],
                        anchor = NW)
            
            array = [roi[0], roi[1], roi[2] + delta_y, roi[3] + delta_x]

        elif clicked_number == 3:   #左下
            rect = self.canvas.create_rectangle(roi[1] + delta_x, roi[0],
                roi[3], roi[2] + delta_y,fill = "green", tags = tags[4],stipple="gray50")#塗りつぶし
        

            upper_left = self.canvas.create_oval(roi[1]+delta_x-5, roi[0]-5,
                        roi[1]+delta_x+5, roi[0]+5, fill="red",
                        width=0, tags = tags[0])
            
            upper_right = self.canvas.create_oval(roi[3]-5, roi[0]-5,
                     roi[3]+5, roi[0]+5, fill="yellow", 
                     width=0, tags = tags[1])

    
            lower_left = self.canvas.create_oval(roi[1]+delta_x-5, roi[2]+delta_y-5,
                        roi[1]+delta_x+5, roi[2]+delta_y+5, fill="blue", 
                        width=0, tags = tags[3])
            
            lower_right = self.canvas.create_oval(roi[3]-5, roi[2]+delta_y-5,
                        roi[3]+5, roi[2]+delta_y+5, fill="green", 
                        width=0, tags = tags[2])

            text = self.canvas.create_text(roi[1]+delta_x, roi[0],
                        text = self.roi_list[self.selected_roi_number].class_name,
                        font = ('FixedSys', 14),
                        tags = tags[5],
                        anchor = NW)

            array = [roi[0], roi[1] + delta_x, roi[2]+delta_y, roi[3]]
        else:
            print("sine")

        changed_roi = np.array(array)

        #ひっくり返ったときの処理
        #→releaseの時に直す

        self.roi_list[self.selected_roi_number].roi = changed_roi

        if not self.roi_list[self.selected_roi_number].keypoint == None:
            self.resize_keypoints(pre_roi = pre_roi, post_roi = changed_roi)

        return


    def resize_keypoints(self, pre_roi, post_roi):
        """
        矩形の拡大縮小に対応したキーポイントの移動
        拡大縮小比に対応した分だけ各キーポイントの位置を変更する
        →各キーポイントの矩形内における相対位置を取得
        →拡大縮小日にかかわらすその比に合わせる
        """
        #print("pre_roi",pre_roi)
        #print("post_roi",post_roi)
        if self.selected_pose_number == None:
            number = self.selected_roi_number
        
        elif self.selected_roi_number == None:
            number = self.selected_pose_number
        
        else:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return



        point_list = self.roi_list[number].keypoint
        vector_list =  self.kyeypoint_vector_list(point_list)
        tags = self.roi_list[number].tags

        pre_x_len = pre_roi[3] - pre_roi[1]
        pre_y_len = pre_roi[2] - pre_roi[0]

        post_x_len = post_roi[3] - post_roi[1]
        post_y_len = post_roi[2] - post_roi[0]

        count = 0
        for point in point_list:
            if count == 0:
                radius = post_x_len * (10 / 100)
            else:
                radius = post_x_len * (5 / 100)
        


            #point/lenの割合(前)
            pre_ratio_x = (point[0] - pre_roi[1]) / pre_x_len
            pre_ratio_y = (point[1] - pre_roi[0]) / pre_y_len

            point[0] = post_roi[1] + pre_ratio_x * post_x_len          
            point[1] = post_roi[0] + pre_ratio_y * post_y_len

            self.canvas.create_oval(point[0]-radius, point[1]-radius,
                        point[0]+radius, point[1]+radius, fill="red",
                        width=0, tags = tags[count+6])  
            count += 1


        self.roi_list[number].keypoint = point_list

        vector_list =  self.kyeypoint_vector_list(point_list)

        for vector in vector_list:
            self.canvas.create_line(vector[1], vector[0], vector[3], vector[2],
                             width = 2.0, fill = 'blue',tags = tags[count + 6])
            count += 1

        #self.roi_list[self.selected_roi_number].keypoint = point_list


        return


    
    def move_roi_instance(self, x, y):
        """
        if tags == None:
            tags = self.tags
        """
        roi = self.roi_list[self.selected_roi_number].roi
        tags = self.roi_list[self.selected_roi_number].tags
        #print("mroi:",tags)
        #copied_tags = copy.copy(tags)
        delta_x = x - self.preX
        delta_y = y - self.preY
        """
        print("preX",self.preX)
        print("preY",self.preY)
        print("x",x)
        print("y",y)
        print("roi",roi)
        print("xahyouX",roi[1] + delta_x)
        print("xahyouY",roi[0] + delta_y)
        print("")
        """

        rect = self.canvas.create_rectangle(roi[1] + delta_x, roi[0] + delta_y,
                roi[3] + delta_x, roi[2] + delta_y,fill = "green", tags = tags[4],stipple="gray50")#塗りつぶし
        

        upper_left = self.canvas.create_oval(roi[1]+delta_x-5, roi[0]+delta_y-5,
                     roi[1]+delta_x+5, roi[0]+delta_y+5, fill="red",
                      width=0, tags = tags[0])

        
        upper_right = self.canvas.create_oval(roi[3]+delta_x-5, roi[0]+delta_y-5,
                     roi[3]+delta_x+5, roi[0]+delta_y+5, fill="yellow", 
                     width=0, tags = tags[1])

   
        lower_left = self.canvas.create_oval(roi[1]+delta_x-5, roi[2]+delta_y-5,
                     roi[1]+delta_x+5, roi[2]+delta_y+5, fill="blue", 
                     width=0, tags = tags[3])
        
        lower_right = self.canvas.create_oval(roi[3]+delta_x-5, roi[2]+delta_y-5,
                     roi[3]+delta_x+5, roi[2]+delta_y+5, fill="green", 
                     width=0, tags = tags[2])

        text = self.canvas.create_text(roi[1]+delta_x, roi[0]+delta_y,
                        text = self.roi_list[self.selected_roi_number].class_name,
                        font = ('FixedSys', 14),
                        tags = tags[5],
                        anchor = NW)

        changed_roi = np.array([roi[0]+delta_y, roi[1]+delta_x ,roi[2]+delta_y ,roi[3]+delta_x])

        self.roi_list[self.selected_roi_number].roi = changed_roi

        if not self.roi_list[self.selected_roi_number].keypoint == None:
            self.move_keypoints(delta_x, delta_y)



        return
    
    def move_keypoints(self, x, y):
        #矩形の移動に姿勢を合わせる
        if not self.selected_roi_number == None:
            number = self.selected_roi_number
        elif not self.selected_pose_number == None:
            number = self.selected_pose_number
        else:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return
        point_list = self.roi_list[number].keypoint
        vector_list =  self.kyeypoint_vector_list(point_list)
        tags = self.roi_list[number].tags
        roi = self.roi_list[number].roi

        count = 0
        x_len = roi[3] - roi[1]
        y_len = roi[2] - roi[0]
        for point in point_list:
            if count == 0:
                radius = x_len * (10 / 100)
            else:
                radius = x_len * (5 / 100)

            
            point[0] += x
            point[1] += y
            self.canvas.create_oval(point[0]-radius, point[1]-radius,
                        point[0]+radius, point[1]+radius, fill="red",
                        width=0, tags = tags[count+6])  
            count += 1



        for vector in vector_list:
            self.canvas.create_line(vector[1]+x, vector[0]+y, vector[3]+x, vector[2]+y,
                             width = 2.0, fill = 'blue',tags = tags[count + 6])
            count += 1

        self.roi_list[number].keypoint = point_list


        return

    def released_select_roi(self,event):
        #矩形を選択

        if self.selected_roi_number == None:
            return
        
        if self.roi_list[self.selected_roi_number].roi[1] > \
                        self.roi_list[self.selected_roi_number].roi[3]: #xがひっくり返ってたら
            
            self.roi_list[self.selected_roi_number].roi[1], \
            self.roi_list[self.selected_roi_number].roi[3] = \
                    self.roi_list[self.selected_roi_number].roi[3], \
                    self.roi_list[self.selected_roi_number].roi[1]
                
        if self.roi_list[self.selected_roi_number].roi[0] > \
                        self.roi_list[self.selected_roi_number].roi[2]: #xがひっくり返ってたら
            
            self.roi_list[self.selected_roi_number].roi[0], \
           self. roi_list[self.selected_roi_number].roi[2] = \
                    self.roi_list[self.selected_roi_number].roi[2], \
                    self.roi_list[self.selected_roi_number].roi[0]
        
        roi = self.roi_list[self.selected_roi_number].roi    #ひっくり返った矩形の再描画

        tags = self.roi_list[self.selected_roi_number].tags
        for tag in tags:
                self.canvas.delete(tag)
        
        rect = self.canvas.create_rectangle(roi[1], roi[0],
                roi[3], roi[2],fill = "green", tags = tags[4],stipple="gray50")
        

        upper_left = self.canvas.create_oval(roi[1]-5, roi[0]-5,
                     roi[1]+5, roi[0]+5, fill="red",
                      width=0, tags = tags[0])

        
        upper_right = self.canvas.create_oval(roi[3]-5, roi[0]-5,
                     roi[3]+5, roi[0]+5, fill="yellow", 
                     width=0, tags = tags[1])

   
        lower_left = self.canvas.create_oval(roi[1]-5, roi[2]-5,
                     roi[1]+5, roi[2]+5, fill="blue", 
                     width=0, tags = tags[3])
        
        lower_right = self.canvas.create_oval(roi[3]-5, roi[2]-5,
                     roi[3]+5, roi[2]+5, fill="green", 
                     width=0, tags = tags[2])

        text = self.canvas.create_text(roi[1], roi[0],
                            text = self.roi_list[self.selected_roi_number].class_name,
                            font = ('FixedSys', 14),
                            tags = tags[5],
                            anchor = NW)
        
        if not self.roi_list[self.selected_roi_number].keypoint == None:
            self.move_keypoints(self.postX - self.preX, self.postY - self.preY)
        


        return
    
    def det_selected_editing_point(self, x, y , pre_mode = "SELECT_ROI"):
        #クリックした編集矩形(姿勢全体)の四隅の決定
        #四隅をクリックしてなければ矩形自体の移動
        """
        self.roi_list = []
        self.class_name_list = []
        self.keypoint_list = []
        self.sizeX = sizeX
        self.sizeY = sizeY
        """
        if self.selected_pose_number == None:
            #編集矩形を表示していなかったら何もしない
            return None, None

        self.change_mode("SELECT_ROI")
        clicked_corner_number = None        #編集矩形の四隅の番号
        #すでに存在する編集矩形の範囲だけ選択すればいい
        roi_object = self.roi_list[self.selected_pose_number]    

        #四隅の決定
        min_x, min_y, max_x, max_y = self.get_pose_editing_roi_corner()     #編集矩形の四隅を決定
        roi = [min_y, min_x, max_y, max_x]


        list0 = [roi[0], roi[1]]    #upper_left [y,x] 赤
        list1 = [roi[0], roi[3]]         #upper_right　黄色 
        
        list2 = [roi[2], roi[3]]         #lower_right　青
        list3 = [roi[2], roi[1]]         #lower_left    緑
        lists = [list0, list1, list2, list3]
        count = 0          
        for l in lists:   #四隅のそれぞれを比較
            if self.corner_is_clicked( l[1], l[0], self.preX, self.preY):
                print("clicked")
                self.canvas.itemconfigure(roi_object.tags[count+35], fill = 'black')
                                #count + 矩形の左上のtag番号
                clicked_corner_number = count  #矩形と角の番号を格納
            else:
                #print("not clicked")
                self.canvas.itemconfigure(roi_object.tags[count+35], fill = 'gray50')

            count += 1
    
        if clicked_corner_number == None:      #四隅をクリックしていない
            #→各キーポイントをクリックしたかを判定→姿勢の編集矩形の四隅以外の位置をクリックしたか判定
            print("NONE editing_oval")
            clicked_keypoint_number = self.det_selected_keypoint(x, y, pre_mode = pre_mode)
             #returnの順番：編集矩形の四隅の番号, 編集矩形(キーポイント)の番号
            return None, clicked_keypoint_number        
            #return self.det_selected_roi(x, y, pre_mode = pre_mode) , None #角の判定が失敗で矩形の判定


        else:      #四隅をクリック
            self.selected_roi_number = None #矩形を選択していないのでself.selected_roi_number はNone
           
            
            self.canvas.delete(self.roi_list[self.selected_pose_number].tags[5])    #コンボボックスを除去
            
            self.canvas.create_text(self.roi_list[self.selected_pose_number].roi[1],
                            self.roi_list[self.selected_pose_number].roi[0],
                            text = self.roi_list[self.selected_pose_number].class_name,
                            font = ('FixedSys', 14),
                            tags = self.roi_list[self.selected_pose_number].tags[5],
                            anchor = NW)            #クラス名の埋め込み
            
            
            
        #returnの順番：編集矩形の四隅の番号, 編集矩形(キーポイント)の番号
        return clicked_corner_number, None


    

    def det_selected_oval(self, x, y, pre_mode = "SELECT_ROI"):
        #クリックした矩形の四隅の決定
        """
        self.roi_list = []
        self.class_name_list = []
        self.keypoint_list = []
        self.sizeX = sizeX
        self.sizeY = sizeY
        """
        #able_list = []
        self.change_mode("SELECT_ROI")
        #i = 0
        #self.selected_roi_number = None             #ここがいけない
        self.selected_corner_number = None
        clicked_obj_number = None
        clicked_corner_number = None
        

        for i in range(0, len(self.roi_list)):
            roi_object = self.roi_list[i]
            roi = roi_object.roi
            list0 = [roi[0], roi[1]]    #upper_left [y,x] 赤
            list1 = [roi[0], roi[3]]         #upper_right　黄色 
            
            list2 = [roi[2], roi[3]]         #lower_right　青
            list3 = [roi[2], roi[1]]         #lower_left    緑
            lists = [list0, list1, list2, list3]
            count = 0
            for l in lists:   #四隅のそれぞれを比較
                if self.corner_is_clicked( l[1], l[0], self.preX, self.preY):
                    print("clicked")
                    self.canvas.itemconfigure(roi_object.tags[count], fill = 'black')
                    
                    clicked_obj_number = i
                    clicked_corner_number = count  #矩形と角の番号を格納
                    #break
                else:
                    #print("not clicked")
                    if count == 0:
                        self.canvas.itemconfigure(roi_object.tags[count], fill = 'red')
                    elif count == 1:
                        self.canvas.itemconfigure(roi_object.tags[count], fill = 'yellow')
                    elif count == 2:
                        self.canvas.itemconfigure(roi_object.tags[count], fill = 'blue')
                    elif count == 3:
                        self.canvas.itemconfigure(roi_object.tags[count], fill = 'green')

                count += 1
        
        if clicked_obj_number == None:
            print("NONE oval", None)
            clicked_pose_number, clicked_obj_number = self.det_selected_pose_keypoint(x, y, pre_mode = pre_mode)
            return clicked_pose_number, clicked_obj_number, None
            #return self.det_selected_roi(x, y, pre_mode = pre_mode) , None #角の判定が失敗で矩形の判定


        else:      
            self.selected_roi_number = clicked_obj_number
            #self.selected_roi.rect["fill"] = "green"
            self.canvas.itemconfigure(self.roi_list[self.selected_roi_number].tags[4],
                                 fill = 'green')
            self.canvas.delete(self.roi_list[self.selected_roi_number].tags[5])
            self.canvas.create_text(self.roi_list[self.selected_roi_number].roi[1],
                            self.roi_list[self.selected_roi_number].roi[0],
                            text = self.roi_list[self.selected_roi_number].class_name,
                            font = ('FixedSys', 14),
                            tags = self.roi_list[self.selected_roi_number].tags[5],
                            anchor = NW)
            
            
            
            
            return None, clicked_obj_number, clicked_corner_number




    
    def corner_is_clicked(self, x, y, clickedX, clickedY, corner_size = 5):
        if clickedX < x-corner_size:
            #print("a")
            return False
        if clickedY < y-corner_size:
            #print("b")
            return False
        if  x + corner_size < clickedX:
            #print("c")
            return False
        if y + corner_size < clickedY:
            #print("d")
            return False
        return True
    
    def vector_is_clicked(self, vector, clickedX, clickedY, width = 5 ):
        if clickedX < vector[1] - width:
           return False
        if clickedY < vector[0] - width:
            return False
        if  vector[3] + width < clickedX:
            return False
        if vector[2] + width < clickedY:
            return False

        return True

    def det_selected_keypoint(self, x, y, pre_mode = "SELECT_ROI"):
        #クリックしたキーポイントの決定
        self.change_mode("SELECT_ROI")
        
        clicked_pose_number = None
        clicked_keypoint_number = None
        

        if self.selected_pose_number == None:
            print("None keypoint in det_selected_keypoint")
            #return:キーポイント番号
            return None
        roi_object = self.roi_list[self.selected_pose_number]
        pose = roi_object.keypoint
        roi = roi_object.roi

        count = 0
        x_len = roi[3] - roi[1]
        for keypoint in pose:   #四隅のそれぞれを比較
            if count == 0:
                corner_size = x_len * (10 / 100)
            else:
                corner_size = x_len * (5 / 100)
            #print("corner_size", corner_size)
                        

            if self.corner_is_clicked( keypoint[0], keypoint[1], x, y, corner_size = corner_size):
                print("clicked")
    
                

              

                self.canvas.itemconfigure(roi_object.tags[4], fill = 'white')

         
                for redder in range(count + 1,len(pose)):
                    #print(redder + 6)
                    self.canvas.itemconfigure(roi_object.tags[redder+6], fill = 'red')
                
                
                #return:キーポイント番号
                for i in range(34, len(roi_object.tags)):
                    self.is_keypint_editing = False
                    self.canvas.delete(roi_object.tags[i])

                if self.selected_keypoint_number == count: 
                    self.keypoint_editing_roi()
                    self.canvas.itemconfigure(roi_object.tags[count+6], fill = 'blue')
                else:
                    self.pose_editing_roi()
                self.selected_keypoint_number = count
                return self.selected_keypoint_number


            else:
                self.canvas.itemconfigure(roi_object.tags[count+6], fill = 'red')
                

            count += 1
        



        #if clicked_obj_number == None:なら
        print("NONE_selected_pose_keypoint", None)
        #self.selected_pose_number = None            #poseを選択していないことにする
        self.is_editing_roi_selected(x, y) #キーポイントの判定が失敗で矩形の判定
        #↾が失敗すると編集矩形が消える
        return None
    
    def is_editing_roi_selected(self, x, y):
        #選択済みの編集矩形がクリックされたかを判定
        x0, y0, x1, y1 = self.get_pose_editing_roi_corner()
        is_clicked = True

        if x < x0:
            is_clicked = False
        if y < y0:
            is_clicked = False
        if x1 < x:
            is_clicked = False
        if y1 < y:
            is_clicked = False
        
        if not is_clicked:
            self.selected_pose_number = None
            #３４番目以降の編集矩形の除去
            for roi_number in range(len(self.roi_list)):
                for i in range(34, 41):     #姿勢とキーポイントの編集矩形
                    self.is_keypint_editing = False
                    self.canvas.delete(self.roi_list[roi_number].tags[i])

        #範囲内をクリックしていればなにもしない
        return





    def det_selected_pose_keypoint(self, x, y, pre_mode = "SELECT_ROI"):
        #クリックしたキーポイントの決定
        self.change_mode("SELECT_ROI")
        
        clicked_pose_number = None
        clicked_keypoint_number = None
        

        for i in range(0, len(self.roi_list)):
            roi_object = self.roi_list[i]
            roi = roi_object.roi
            pose = roi_object.keypoint
            if pose == None:
                continue

            count = 0
            x_len = roi[3] - roi[1]
            for keypoint in pose:   #四隅のそれぞれを比較
                if count == 0:
                    corner_size = x_len * (10 / 100)
                else:
                    corner_size = x_len * (5 / 100)
                #print("corner_size", corner_size)
                           

                if self.corner_is_clicked( keypoint[0], keypoint[1], x, y, corner_size = corner_size):
                    #def corner_is_clicked(self, x, y, clickedX, clickedY, corner_size = 5):
                    print("clicked")
                    self.canvas.itemconfigure(roi_object.tags[count+6], fill = 'black')
                    
                    print("self.selected_pose_number", self.selected_pose_number)
                    print("i", i)
                    """
                    if self.selected_pose_number == i: #姿勢選択中にキーポイントをクリック
                        #print("count",count)
                        self.selected_keypoint_number = count 
                    else:                                       #初めて姿勢クリック
                        #print("sel_None")
                        self.selected_keypoint_number = None
                    """
                    self.selected_keypoint_number = count
                    self.selected_pose_number = i
                    
                    #clicked_keypoint_number = count  #矩形と角の番号を格納
                    #self.selected_roi_number = i
                    #break
                    self.canvas.itemconfigure(roi_object.tags[4], fill = 'white')
                    self.selected_roi_number = None
                    #print("self.selected_keypoint_number",self.selected_keypoint_number)
                    """
                    if self.selected_keypoint_number == None:
                        print("pose")
                   
                        self.pose_editing_roi()
                    else:
                        print("keypoint")
                        for redder in range(count + 1,len(pose)):
                            #print(redder + 6)
                            self.canvas.itemconfigure(roi_object.tags[redder+6], fill = 'red')
                        self.keypoint_editing_roi()
                    """
                    for redder in range(count + 1,len(pose)):
                        #print(redder + 6)
                        self.canvas.itemconfigure(roi_object.tags[redder+6], fill = 'red')
                    self.pose_editing_roi()
                    return i, None
                else:
                    self.canvas.itemconfigure(roi_object.tags[count+6], fill = 'red')
                    

                count += 1
            
            vector_list = self.kyeypoint_vector_list(pose)
            for vector in vector_list:   #四隅のそれぞれを比較
                if self.vector_is_clicked( vector, self.preX, self.preY):
                    print("vector-clicked")
                    self.canvas.itemconfigure(roi_object.tags[count+6], fill = 'black')
                    
                    
                    self.selected_pose_number = i
                    
                    #clicked_keypoint_number = count  #矩形と角の番号を格納
                    #self.selected_roi_number = i
                    #break
                    self.canvas.itemconfigure(roi_object.tags[4], fill = 'white')
                    self.selected_roi_number = None


                    self.pose_editing_roi()

                    return i, None
                    
                    
                    
                    return i, None
                
                else:
                    self.canvas.itemconfigure(roi_object.tags[count+6], fill = 'blue')

                count += 1


        #if clicked_obj_number == None:なら
        print("NONE_selected_pose_keypoint", None)
        self.selected_pose_number = None
        return None, self.det_selected_roi(x, y, pre_mode = pre_mode) #角の判定が失敗で矩形の判定




    
    def det_selected_roi(self, x, y, pre_mode = "SELECT_ROI"):
        #クリックした矩形の決定
        """
        self.roi_list = []
        self.class_name_list = []
        self.keypoint_list = []
        self.sizeX = sizeX
        self.sizeY = sizeY
        """
        #able_list = []
        self.change_mode("SELECT_ROI")
        self.selected_roi_number = None
        clicked = []

    
        for i in range(0,len(self.roi_list)):
            roi_object = self.roi_list[i]
            roi = roi_object.roi
            self.canvas.delete(roi_object.tags[5])
            if x < roi[1]:
                #print("s")  
                self.canvas.itemconfigure(roi_object.tags[4], fill = 'white')
                self.canvas.create_text(roi[1], roi[0],
                                    text = roi_object.class_name,
                                    font = ('FixedSys', 14),
                                    tags = roi_object.tags[5],
                                    anchor = NW)
                
                continue
            if y < roi[0]:
                #print("d")
                self.canvas.itemconfigure(roi_object.tags[4], fill = 'white')
                
                self.canvas.create_text(roi[1], roi[0],
                                    text = roi_object.class_name,
                                    font = ('FixedSys', 14),
                                    tags = roi_object.tags[5],
                                    anchor = NW)
                
                continue
            if  roi[3] < x:
                #print("f")
                self.canvas.itemconfigure(roi_object.tags[4], fill = 'white')
                
                self.canvas.create_text(roi[1], roi[0],
                                    text = roi_object.class_name,
                                    font = ('FixedSys', 14),
                                    tags = roi_object.tags[5],
                                    anchor = NW)
                
                continue
            if roi[2] < y:
                #print("g")
                self.canvas.itemconfigure(roi_object.tags[4], fill = 'white')
                
                self.canvas.create_text(roi[1], roi[0],
                                    text = roi_object.class_name,
                                    font = ('FixedSys', 14),
                                    tags = roi_object.tags[5],
                                    anchor = NW)
                
                continue
            self.canvas.itemconfigure(roi_object.tags[4], fill = 'white')
            
            self.canvas.create_text(roi[1], roi[0],
                                    text = roi_object.class_name,
                                    font = ('FixedSys', 14),
                                    tags = roi_object.tags[5],
                                    anchor = NW)
            
            clicked.append(i) #インスタンスと要素番号
            #print("tags of roi_object",roi_object.tags)
            
      
  
        if pre_mode == "CLASS_NAME":
            self.change_mode("CLASS_NAME")
        
        if len(clicked) == 0:
                self.change_mode(pre_mode)
                print(None)
                return None
        
        min_area = self.sizeX * self.sizeY  #クリック範囲内の矩形の面積の最小値
        min_obj = None #↾に対応するROIインスタンス
        for j in clicked:
            area = self.det_area(self.roi_list[j].roi)
            #print(area)
            if area < min_area:
                min_area = area
                min_obj = j
        #print(min_obj)
        #print("tags of min_obj",min_obj.tags)
        self.selected_roi_number = min_obj


        self.canvas.itemconfigure(self.roi_list[self.selected_roi_number].tags[4], 
                                fill = 'green')
        
        self.canvas.create_text(self.roi_list[self.selected_roi_number].roi[1], 
                    self.roi_list[self.selected_roi_number].roi[0],
                    text = self.roi_list[self.selected_roi_number].class_name,
                    font = ('FixedSys', 14),
                    tags = self.roi_list[self.selected_roi_number].tags[5],
                    anchor = NW)

        
        return self.selected_roi_number


    
    def det_area(self, array):
        return(array[3] - array[1]) * (array[2] - array[0])
    
    
    def define_image_size(self):
        #画像サイズの決定or変更
        #filewin = tkinter.Toplevel(self.master)
        #button = tkinter.Button(filewin, text="Do nothing button")
        #button.pack()
        #縦横比を入力するウィンドウを生成(整数)

        size_def = tkinter.Tk()
        size_def.title("キャンバスの大きさ")
        size_def.geometry("650x100")
        size_window = Size_window(main_window = self, master = size_def)
        size_window.mainloop()
 
        return
    def change_mode(self, mode, is_first = False):
        self.mode = mode
        print("mode:{}".format(mode))
        
        if not is_first:        #mode_labelの宣言前に呼び出されるときのエラーを回避
            self.mode_label["text"] = "mode:{}".format(mode)
 
        return mode


    def define_roi(self, roi = None):
        #矩形サイズの決定or変更
        self.change_mode("ROI")

        return
    
    def define_class_name(self, class_name = None):
        #クラス名を宣言する
        self.change_mode("CLASS_NAME")
        return

    def change_keypoint(self, keypoint = None):
        #キーポイントを変更する
        self.change_mode("KEYPOINT")

        return
    def change_roi(self):
        self.change_mode("SELECT_ROI")
        return

    def save_data(self):    #セーブ
        #キーポイント判定(人以外の点にキーポイントを入力していたら削除する)
        

        #クラス判定(クラス名を入力していないものがあったら排除する)
        for roi_instance in self.roi_list:
            if roi_instance.class_name == None:
                print("クラスが入力されていない矩形があります")
                return



        image = np.full((self.sizeY, self.sizeX, 3), 0)   #白
        roi = []
        classID = []
        score = []
       


        """
                            "tag6_" + str(self.tag_number),     #キーポイント：鼻
                    "tag7_" + str(self.tag_number),     #キーポイント：右肩
                    "tag8_" + str(self.tag_number),     #キーポイント：左肩
                    "tag9_" + str(self.tag_number),     #キーポイント：喉
                    "tag10_" + str(self.tag_number),    #キーポイント：右ひじ 
                    "tag11_" + str(self.tag_number),    #キーポイント：左ひじ
                    "tag12_" + str(self.tag_number),    #キーポイント：右手首
                    "tag13_" + str(self.tag_number),    #キーポイント：左手首
                    "tag14_" + str(self.tag_number),    #キーポイント：右尻
                    "tag15_" + str(self.tag_number),    #キーポイント：左尻
                    "tag16_" + str(self.tag_number),    #キーポイント：右ひざ
                    "tag17_" + str(self.tag_number),    #キーポイント：左ひざ
                    "tag18_" + str(self.tag_number),    #キーポイント：右くるぶし
                    "tag19_" + str(self.tag_number),    #キーポイント：左くるぶし


          PART_STR = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
                "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
                "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
   
        """
        person_list = []
        not_person_list = []
        person_keypoints = []
        not_person_keypoints = []
        #キーポイントの編集と並び替え
        for roi_obj in self.roi_list:
            if self.class_names.index(roi_obj.class_name) == 1:        #クラスが人なら
      

                person_list.append(roi_obj)
                if roi_obj.keypoint == None:
                    person_keypoints.append(np.zeros((17,3), dtype = np.float32))
                else:


                    count = 0
                    pose = []
                    for keypoint in roi_obj.keypoint:
                        #キーポイントを整形
                        if count == 3:      #喉のキーポイント
                            count += 1
                            continue
                        point = np.array([keypoint[0], keypoint[1], 1])    #各キーポイントの値
                        pose.append(point)
                        if count == 0:      #鼻の格納が終わったら
                            point_token = np.array([0, 0, 0])    #使用しない値(両目、両耳)の格納
                            pose.append(point_token)
                            pose.append(point_token)
                            pose.append(point_token)
                            pose.append(point_token)
                

                
                        count += 1
                
                    person_keypoints.append(np.array(pose))
            else:
                not_person_list.append(roi_obj)
                not_person_keypoints.append(np.zeros((17,3), dtype = np.float32))



                

        
        keypoints = person_keypoints + not_person_keypoints
        instance_list = person_list + not_person_list       #人クラスを前にして統合

        


        for roi_obj in instance_list:
            roi.append(roi_obj.roi)
            classID.append(self.class_names.index(roi_obj.class_name))
            score.append(1.0)



        roi = np.array(roi)
        classID = np.array(classID)
        score = np.array(score)
        keypoints = np.array(keypoints)
        
        print("ファイル名を入力してください")
        name = input()
        file_name = "kouzudata_{}".format(name)
        
        #人のクラスが最初に来るように並び替え
        #print("roi", roi)
        #print("classID", classID)
        #print("score", score)
        #print("keypoint", keypoints)


      
        #self.log("roi", roi)
        #self.log("classID", classID)
        #self.log("score", score)
        #self.log("keypoint", keypoints)
        #print("roi_type", type(roi))
        #print("classID_type", type(classID))
        #print("score_type", type(score))
        #print("keypoint_type", type(keypoints)
        #print("class", classID)
        #classID = np.array(classID)
        #print("classID_type", type(classID))

        
        

        saved_image =  Keypoint_result(image, roi, keypoints, classID, score, file_name)
     

        #saved_image = ImageData(image, roi, classID, score, file_name)
        
        #os.chdir('./pickle_keypoint')    #picklesフォルダに移動
        with open("./kouzu_data/{}.pickle".format(file_name), 'wb') as f:
            pickle.dump(saved_image, f)
        #os.chdir('../')

        print("saveEnd")

        #self.search(file_name + ".pickle")
        #self.visualize_saved_data("{}.pickle".format(file_name))
        

        return
    
    def visualize_saved_data(self, file_name):
        vi = Visualize_IoU(file_name)
        
        vi.while_token(times = 1, size_range = 10000, roi_range = 10000, keypoint_range = 0.0, 
                            roi_move_range = 100, append_times = 0, remove_times = 0, way_of_compute = "vector5")
        #vi.only_keypoint()


        return
    def log(self, text, array=None):
        """Prints a text message. And, optionally, if a Numpy array is provided it
        prints it's shape, min, and max values.
        """
        if array is not None:
            text = text.ljust(25)
            text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
                str(array.shape),
                array.min() if array.size else "",
                array.max() if array.size else ""))
        print(text)
        return
    
    def search(self, made_data_name):
        way_of_compute = "vector3"
        search_sort = Search_sort(search_image = made_data_name, way_of_compute = way_of_compute)
        start = time.time()
        pickle_list = search_sort.sort(way_of_compute = way_of_compute)
        end = time.time()
        print("serch_time:{}".format(end - start))
        
        #類似度最大のものを表示
        visualize_name = pickle_list[-1][0].file_name + ".pickle"
        vi = Visualize_IoU(visualize_name)
        vi.while_token(times = 0, size_range = 1000, roi_range = 1000,
                         append_times = 0, remove_times = 0, way_of_compute = "vector5")


        return

    def click(self, event):     #クリックされたときに呼び出される関数
        # クリックされた場所に描画する
        #クラス名入力直後なら、クラス名を決定する
        
        #if self.after_det_class_name:
        #    self.get_class_name_from_box()

        if not self.selected_roi_number == None:    #コンボボックスをキャンバス外にどける
            self.roi_list[self.selected_roi_number].combo_box.place(x = -100, y = -100)
            #クラス名の確定

        
        self.preX = event.x
        self.preY = event.y
        """
        #何もないところクリック→ROIモード
        #矩形内クリック→選択モード（移動など）
           #右クリックorボタンでキーポイント？ 
        #四隅クリック→選択モード(サイズ変更))
        """

        """
        ROIもーど
            どこクリックしても新規ROI
        selectモード
            矩形クリック→矩形移動
            矩形四隅クリック→矩形拡大縮小
            余白クリック→ズーム時の表示位置変更

        """
        if self.mode == "ROI":
            self.clicked_roi(event)
        elif self.mode == "SELECT_ROI":
            """
            クリック判定(優先順)
            編集矩形
                編集矩形
                矢印
                4隅
            矩形の四隅
            矩形
            姿勢
            キーポイント
            
            """

            #編集矩形ガス銭存在し，それをクリックしたかどうか
            self.clicked_editing_corner_number, self.clicked_editing_keypoint_number = \
                                   self.det_selected_editing_point(event.x, event.y)
            #編集矩形すらクリックされてなければself.selected_pose_number = Noneになってる
            #clicked_editing_corner_number:編集矩形の四隅の番号
            #clicked_editing_keypoint_number:編集矩形(キーポイント)の番号

            if not self.clicked_editing_corner_number == None:
                #編集矩形の四隅がクリックされていれば
                #print("corner")
                return
            elif not self.clicked_editing_keypoint_number == None:
                #編集矩形内のキーポイントがクリックされていれば
                #print("key")
                return
            elif not self.selected_pose_number == None:
                #編集矩形自体がクリックされていれば
                #print("editing_roi")
                return
            else:
                print("\nself.clicked_editing_corner_number", self.clicked_editing_corner_number)
                print("self.clicked_editing_keypoint_number",self.clicked_editing_keypoint_number)
                print("self.selected_pose_number", self.selected_pose_number)
                
                #print("noEditing")


                #どれもクリックされていなければ
                #aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
                is_pose, is_rect, is_oval = self.det_selected_oval(event.x, event.y, pre_mode = "SELECT_ROI") #矩形か四隅をクリックしたかどうか
                print("is_rect", is_rect)
                print("is_pose", is_pose)
                if is_rect == None and is_pose == None:     #余白をクリック
                    #print("rectNone")
                    self.change_mode("ROI")
                    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                    #self.clicked_roi(event)
                    #self.change_camera(event)
                else:                    #余白以外をクリック
                    if abs(self.preY - self. postY) == 0 and \
                        abs(self.preX - self. postX) == 0:      #ダブルクリック
                        self.add_class_name()
                    #print("rectis")


                    self.selected_pose_number = is_pose
                    self.selected_corner_number = is_oval
                    self.selected_roi_number = is_rect
                    #self.clicked_select_roi(event) #矩形をクリックされたときの処理


        """
        if self.mode == "ROI":
            self.clicked_roi(event)
        elif self.mode == "CLASS_NAME":
            self.clicked_class_name(event)
        elif self.mode == "KEYPOINT":
            self.clicked_keypoint(event)
        elif self.mode == "SELECT_ROI":
            self.clicked_select_roi(event)
        else:
            print("INVALID MODE*{}".format(self.mode))
        """

        return
    
    def release(self, event):
        # クリックされた場所に描画する
        self.postX = event.x
        self.postY = event.y

        if self.mode == "ROI":
            self.released_roi(event)
        elif self.mode == "CLASS_NAME":
            self.released_class_name(event)
        elif self.mode == "KEYPOINT":
            self.released_keypoint(event)
        elif self.mode == "SELECT_ROI":
            self.released_select_roi(event)
        
        
        return

    
    def motion(self, event):
        # クリックされた場所に描画する
        if self.mode == "ROI":
            self.dragged_roi(event)
        elif self.mode == "CLASS_NAME":
            self.dragged_class_name(event)
        elif self.mode == "KEYPOINT":
            self.draggeded_keypoint(event)
        elif self.mode == "SELECT_ROI":
            self.dragged_select_roi(event)
        
        
        return
    
    
    def click_left(self, event):
        # クリックされた場所に描画する
        self.preX = event.x
        self.preY = event.y
        """
        #何もないところクリック→ROIモード
        #矩形内クリック→選択モード（移動など）
           #右クリックorボタンでキーポイント？ 
        #四隅クリック→選択モード(サイズ変更))
        """
        is_pose, is_rect, is_oval=  self.det_selected_oval(event.x, event.y, pre_mode = "ROI")

        if is_rect == None:     #何もないところクリック
            self.plain_menu(event)
        else:
            #if is_oval == None:    #矩形かコーナーをクリック
            self.roi_menu(event)



        return
    
    def release_left(self, event):
        # クリックされた場所に描画する
        self.postX = event.x
        self.postY = event.y
        if self.mode == "ROI":
            self.released_roi(event)
        elif self.mode == "CLASS_NAME":
            self.released_class_name(event)
        elif self.mode == "KEYPOINT":
            self.released_keypoint(event)
        elif self.mode == "SELECT_ROI":
            self.released_select_roi(event)
        
        
        return

    
    def motion_left(self, event):
        # クリックされた場所に描画する
        if self.mode == "ROI":
            self.dragged_roi(event)
        elif self.mode == "CLASS_NAME":
            self.dragged_class_name(event)
        elif self.mode == "KEYPOINT":
            self.draggeded_keypoint(event)
        elif self.mode == "SELECT_ROI":
            self.dragged_select_roi(event)
        
        
        return



def main():
    root = tkinter.Tk()
    sizex = 800
    sizey = 600
    root.geometry("{}x{}".format(sizex, sizey))
    app = Application(master=root, sizeX = sizex, sizeY = sizey)
    app.mainloop()



    return


if __name__ == "__main__" :
    main()
"""
問題点
レイアウトがなんかおかしい
なんかちいさくね
矩形に重なった位置に新しい矩形をかけない
クラス名保存のタイミングがおかしい

"""



from scipy import io


from loadAndComparison import Keypoint_result
from loadAndComparison import LoadAndComparison
from loadAndComparison import ImageData

import IoU_getter
from IoU_getter import Get_IoU

import numpy as np





def main():
    keypoint_index = [15, 13, 11, 12, 14, 16, -1, -1, -1, -1, 9, 7, 5, 6, 8, 10]

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

    #鼻，両目，両耳

    get_iou = Get_IoU(class_names)



    lac =  LoadAndComparison()

    mat = io.loadmat("mpii_human_pose_v1_u12_1.mat")

    
    #print("mat['RELEASE']['annolist'][0, 0][0]:",mat['RELEASE']['annolist'][0, 0][0])
    #i = 0
    #while len(mat['RELEASE']['annolist'][0, 0]) < i
    #print("anno_len:", len(mat['RELEASE']['annolist'][0, 0][0]))
    count = 0
    for anno in mat['RELEASE']['annolist'][0, 0][0]:        #[0 ,0]でおｋ？
                                    

        img_fn = anno['image']['name'][0, 0][0]         #ファイル名
        
        
        


        try:
            pickle_file = lac.load_keypoint("{}__keypoint__.pickle".format(img_fn), is_display = False)
        except FileNotFoundError:
            continue
        
        lac.cv2_display_keypoint(pickle_file.image,
                                    pickle_file.roi,
                                    pickle_file.keypoint,
                                    pickle_file.classID,
                                    pickle_file.score,
                                    class_names)
        
        print("a")
        #print("sssssssssssssssssssssssssssssssssssssssssssssss")
    
        
        delete_list = []
        for i in range(len(pickle_file.classID)):
            if pickle_file.classID[i] != 1:
                continue

            delete_list.append(i)
        
        pickle_file.roi =  [e for i,e in enumerate(pickle_file.roi) if i not in delete_list]
        pickle_file.score = [e for i,e in enumerate(pickle_file.score) if i not in delete_list]
        pickle_file.classID = [e for i,e in enumerate(pickle_file.classID) if i not in delete_list]
        pickle_file.keypoint = [e for i,e in enumerate(pickle_file.keypoint) if i not in delete_list]
        person_roi = [e for i,e in enumerate(pickle_file.roi) if i in delete_list]
        #person_classID = [e for i,e in enumerate(pickle_file.keypoint) if i in delete_list]
        person_score = [e for i,e in enumerate(pickle_file.score) if i in delete_list]
                    #人の矩形のデータを削除
        
        


        #print("anno['annorect']:\n", anno['annorect'])
  
        #print("str(anno['annorect'].dtype):\n", str(anno['annorect'].dtype))

    
        #print("anno\n:",anno)
  
        
        if 'annopoints' in str(anno['annorect'].dtype):
            count += 1
            
            #print("anno['annorect']['annopoints']:\n", anno['annorect']['annopoints'])
            #print("anno['annorect']:\n", anno['annorect'])

            annopoints = anno['annorect']['annopoints'][0]
            #print("anno['annorect'][0, 0]['annopoints']\n",anno['annorect'][0]['annopoints'])
            #print("anno['annorect'][0, 1]['annopoints']\n",anno['annorect'][1]['annopoints'])   
            #print("anno['annorect'][0,2]['annopoints']\n",anno['annorect'][2]['annopoints'])            
            
            #print("anno['annorect']['annopoints']\n",anno['annorect']['annopoints'])            
            #print("anno['annorect']['annopoints'][0]\n",anno['annorect']['annopoints'][0])

            #point = anno['annorect']['point']
            roi = []
            classID = []
            score = []
            keypoint = []
          

            for annopoint in annopoints:
                #print("point:", annopoint['point'])

                #print(pickle_file.sizeX, pickle_file.sizeY)
                if annopoint.size:
                    # joint coordinates
                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]


                    # visibility list
                    
                    if 'is_visible' in str(annopoint.dtype):
                        is_visible = [v[0] if v.size > 0 else [0] for v in annopoint['is_visible'][0]]
                    else:
                        is_visible = None
                    
                    joint_pos = np.zeros((17, 3), dtype = np.int)
                    upper_neck = np.array([0, 0, 0])
                    head_top = np.array([0, 0, 0])
                    for _j_id, (_x, _y, _is_visible) in zip(j_id, zip(x, y, is_visible)):
                        _j_id = int(_j_id)
                        _is_visible = int(_is_visible[0]) if len(_is_visible) > 0 else _is_visible
                        if keypoint_index[_j_id] == -1:
                            if _j_id == 8:
                                upper_neck = np.array([int(_x), int(_y), int(_is_visible)])
                            elif _j_id == 9:
                                head_top = np.array([int(_x), int(_y), int(_is_visible)])
                            continue

                        joint_pos[keypoint_index[_j_id]] = np.array([int(_x), int(_y), int(_is_visible)])
                                
                    noseX = (head_top[0] + upper_neck[0]) / 2
                    noseY = (head_top[1] + upper_neck[1]) / 2
                    nose_is_visible = head_top[2] * upper_neck[2]
                    joint_pos[0] = np.array([int(noseX), int(noseY), int(nose_is_visible)])



                    
                    roi_number, keypoint_roi = det_corres_roi(joint_pos, person_roi, get_iou)      #キーポイントに対応するROIを決定
            
            keypoint.append(joint_pos)
            classID.append(1)
            #score.append(1.0)
            #roi.append(np.array([y1, x1, y2, x2]))
            
            if not roi_number == None:  
                score.append(person_score.pop(roi_number))
                roi.append(person_roi.pop(roi_number))
            else:
                score.append(1.0)
                roi.append(keypoint_roi)
            






            pickle_file.keypoint = np.array(keypoint + pickle_file.keypoint)    #キーポイントの矩形を作成
            pickle_file.classID = np.array(classID + pickle_file.classID)
            pickle_file.score = np.array(score + pickle_file.score)
            pickle_file.roi = np.array(roi + pickle_file.roi)
            

            lac.cv2_display_keypoint(pickle_file.image,
                                    pickle_file.roi,
                                    pickle_file.keypoint,
                                    pickle_file.classID,
                                    pickle_file.score,
                                    class_names)
            #print(len(keypoint))
            
                        
    print(count)
    return
    #keypoint_index = [15, 13, 11, 12, 14, 16, -1, -1, -1, -1, 9, 7, 5, 6, 8, 10]
def det_corres_roi(joint_pos, person_rois, get_iou):
    #joint_posに対応するROIをperson_roiから選択する
    roi_number = None
    min_x = 10000
    min_y = 10000
    max_x = 0
    max_y = 0
    
    for joint in joint_pos:
        #joint_posを完全に覆う矩形を決定
        if joint[0] == 0 and joint[1] == 0:
            continue
        
        if joint[2] == 0:
            continue

        print(joint)
        if joint[0] < min_x:
            min_x = joint[0]
        if joint[1] < min_y:
            min_y = joint[1]
        if joint[0] > max_x:
            max_x = joint[0]
        if joint[1] > max_y:
            max_y = joint[1]
        

    keypoint_roi = np.array([min_y, min_x, max_y, max_x])   ##joint_posを完全に覆う矩形を決定
    
    print("keypoint_roi", keypoint_roi)


    count = 0
    max_IOU = 0
    #person_roisのそれぞれの矩形とkeypoint_roiを比較，keypoint_roiをすっぽり覆うもののリストを取得
    for person_roi in person_rois:
        print("person_roi", person_roi)
        iou = get_iou.iou(person_roi, keypoint_roi)
        print("iou:",iou)
        if iou > max_IOU: #iouが現在の最大値を上回る
            print("max_iou exceeded")
            roi_number = count
        
        count += 1

    if max_IOU < 0.7:
        print("iou under  sleshhold")
        roi_number = None

    
    """
    candidate_rois = []
    candidate_rois_index = []        #candidate_roiの添え字
    count = 0
    for person_roi in person_rois:
        print("p_roi", person_roi)
    
        if keypoint_roi[0] > person_roi[0]:
            count += 1
            continue
        if keypoint_roi[1] > person_roi[1]:
            count += 1
            continue
        if keypoint_roi[2] < person_roi[2]:
            count += 1
            continue
        if keypoint_roi[3] < person_roi[3]:
            count += 1
            continue
        
        candidate_rois.append(person_roi)
        candidate_rois_index.append(count)
        count += 1
    
    max_IOU = 0
    
    # candidate_roiの中からkeypoint_roiとのIoUが最大のものを選択する．
    for candidate_roi, candidate_roi_index in zip(candidate_rois, candidate_rois_index):
        print("iou",get_iou.iou(person_roi, keypoint_roi))
        if get_iou.iou(person_roi, keypoint_roi) > max_IOU: #iouが現在の最大値を上回る
            print("a")
            roi_number = candidate_roi_index
    """



    return roi_number, keypoint_roi



"""
     0.鼻           
    1.右目          
    2.左目          
    3.右耳          
    4.左耳          
    5.右肩          
    6.左肩          
    7.右ひじ        
    8.左ひじ        
    9.右手首        
    10.左手首       
    11.右尻         
    12.左尻         
    13.右ひざ       
    14.左ひざ       
    15.右くるぶし   
    16.左くるぶし   
"""
if __name__ == "__main__":
    main()



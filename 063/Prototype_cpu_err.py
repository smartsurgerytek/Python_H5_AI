#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
# from tensorflow import keras
# import segmentation_models as sm
import os
import os.path as path # add by charley
import cv2  #( cv2 pip install opencv_python==4.2.0.34)
import numpy as np
from lib.predict_image import *
ckptname='test'  #最佳CKPT編號 1110830 先給預設值
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# sm.set_framework('tf.keras')
# sm.framework()
# print("TF version: {}.".format(tf.__version__))
# print("Keras version: {}.".format(keras.__version__))

# 取得目前檔案的位置
from lib.cwdS import Cwd_path
# mazda =Cwd_path()
# print(mazda)
#cwd=str(mazda)


            
def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

#'''
if __name__ == "__main__":
    # B, G, R
    # correct_label_target": [1"Adipose_tissue", 2"Articular_capsule", 3"Bipolar_radio_frequency", 4"Bleeding_point", 5"Bone_curettes", 
    # 6"Burr", 7"Drainage_tube", 8"Foreign_material", 9"Hook_probe", 10"Intervertebral_disc", 
    # 11"Intervertebral_disc_after_injection", 12"Kerrison_punch", 13"Lamina", 14"Ligamentum_flava", 15"Needle", 
    # 16"Organ_tissue", 17"Probe", 18"Proforma", 19"Rongeur", 20"Scalpel", 
    # 21"Spinal_nerve", 22"Suspected_Spinal_nerve", 23"Suspected_intervertebral_disc"]}
    color_label = {
          0:[255,255,255],#Alveolar_bone
          1:[65,127,0],#'Caries'
          2:[0,240,255],#Alveolar_bone
          3:[0,0,0],#'Caries'
    }
#    "Intervertebral_disc", "Spinal_nerve"
    text_label = {
        0:"BG",
        1:"Intervertebral_disc",  
        2:"Spinal_nerve",   
                 }
    pos_idx = [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4)]#顯示標註的起位置
#'''            
# solve_cudnn_error()

#os.chdir('D:\\HomeWork\\AI_PY\\055') #指定預設的開啟位置
#cwd = os.getcwd()
cwd = str(Cwd_path())   #開啟檔案目前開啟位置
print('cwd :')
print(cwd)
print('str(cwd) :')
print(str(cwd))
# print(cwd) F:\SurgeryAnalytics\AI_Cases\Dentistry3
print("cwd =",cwd)
AI_Path = cwd #cwd.split('\\C_pred')[0]
#AI_Path0 = cwd.split('\\C_pred')[0]
#AI_Path00 = "F:\SurgeryAnalytics\AI_Cases\Dentistry3"
#print(AI_Path00)
print("AI_Path  { } ",AI_Path)

#p = path.dirname(AI_Path)                                       #抓取目錄名稱#不可用
#print("抓取目錄名稱 p = os.path.dirname() =", p)                #不可用
#r = path.join(p, '\\B_AI_training\\Dentistry_config.json')      #不可用                #將目錄與檔名結合 
#print("將目錄與檔名結合 os.path.join(p,f) =", r)                 #不可用 

# AI_Data_jsonFile=AI_Path+'\\B_AI_training\\Dentistry_config.json'
AI_Data_jsonFile=path.join(AI_Path, 'Dentistry_config.json') 
print(AI_Data_jsonFile)


import json
#jsonFile= r'D:\HomeWork\AI_PY\055\Dentistry_config.json'
# jsonFile= r'D:\SurgeryAnalytics\AI_Cases\Dentistry3\B_AI_training\Dentistry_config.json'
jsonFile=AI_Data_jsonFile
with open(jsonFile,"r") as f:
    config = json.load(f)

#config
#print(jsonFile)

# import os
import time
time.localtime()
train_time=time.strftime("%Y%m%d%H%M%S", time.localtime())

#cwd = os.getcwd()
DATA_DIR = config["split_folder_path"]# 轉好檔的標記資料路徑(label為mask(即.png格式))
x_test_dir = os.path.join(DATA_DIR, 'test', 'img')
y_test_dir = os.path.join(DATA_DIR, 'test', 'label')
img_folder_path = x_test_dir # 要預測的圖片資料夾路徑

#save_pred_img_folder_path = r'D:\HomeWork\AI_PY\055\prediction'+ "/"+train_time # 儲存預測圖片的根目錄資料夾路徑
save_pred_img_folder_path = path.join(AI_Path, 'prediction'+ "/"+train_time)
if not os.path.isdir(save_pred_img_folder_path):
    os.makedirs(save_pred_img_folder_path)
    
#增加預測對照圖的顯示位置1110831 add by charley
save_compare_img_folder_path = r'compare_output'+ "/"+train_time # 儲存預測對照圖的根目錄資料夾路徑 
if not os.path.isdir(save_compare_img_folder_path):
    os.makedirs(save_compare_img_folder_path)
#增加預測對照圖的顯示位置1110831 add by charley    
    
model_name = config["model"] # 使用的模型名稱 (Unet/FPN/Linknet/PSPNet)
backbone_name = config["BACKBONE"]# 使用的backbone名稱
num_classes = config["num_classes"] # 模型預測的類別數量, 此數值要+1(for unlabel)
print(num_classes)
color_list = [
                [0,240,255],#Alveolar_bone
                [65,127,0],#'Caries', 
             ]
                # 針對資料前處理的correct_label清單, 依序填入想要顯示的BGR色碼 (註:是BGR不是RGB), background顏色不用填.

data_height=config["data_height"]
data_width=config["data_width"]

# MODEL=model_name
# if MODEL == 'PSPNet':
#     data_height=384
#     data_width=384
    # tf.compat.v1.disable_eager_execution()

ckpt_folder_path = config["save_ckpt_Folder"] # 存放model weight的資料夾路徑
# ckpt_folder_path="F:\\SurgeryAnalytics\\AI_Cases\\Dentistry/1_Linknet_efficientnetb7_0_1/B_AI_training/ckpt/20220712171222"
print(ckpt_folder_path)
config["save_pred_img_folder_path"]  = f"{cwd}/{save_pred_img_folder_path}"
print(save_pred_img_folder_path) 

#save_compare_img_folder_path
config["save_compare_img_folder_path"]  = f"{cwd}/{save_compare_img_folder_path}"
print(save_compare_img_folder_path)      #增加預測對照圖的顯示位置1110831 add by charley
config["color_list"] = color_list
with open(jsonFile,"w") as f:
    json.dump(config, f)

print(config)

# import os.path as path     # add by charley 1110830
import json                  # add by charley 1110830
import operator              # add by charley 1110830
# def ckptnmae():
#     #print(ckpt_folder_path)
#      dic = {}
#      for ckptFileName in os.listdir(ckpt_folder_path):
#          #print(ckptFileName.split('_')[3]) # 列示全部的ckpt值
#          k = ckptFileName.split('_')[0].strip() # key trim
#          dic[k] = ckptFileName.split('_')[1].strip()  # value trim
#      print(dic)  # 列示全部的ckpt值    
#      json_object = json.dumps(dic, indent=4)
#      max_keys = [key for key, value in dic.items() if value == max(dic.values())]
#      print(max_keys[0]) #如為多個找出其中一個即可
#      print("best solution ckpt-{} !".format(max_keys[0]))
#      ckpname=max_keys[0] #存入自訂變數
#      return ckpname

# AI_Cases/Dentistry2/C_pred/compare_output
# AI_Cases/Dentistry2/C_pred/labels
# import shutil
# def delfolder_compare_output():
#     pathTeststr = AI_Path +"/C_pred/compare_output"
#     print(pathTeststr)
#     #pathTest = r pathTeststr
#     #os.remove(path)
#     try:
#         shutil.rmtree(pathTeststr)
#     except OSError as e:
#         print(e)
#     else:
#         print("The directory is deleted successfully")
    
# def delfolder_compare_labels():
#     pathTeststr = AI_Path +"/C_pred/labels"
#     print(pathTeststr)
#     #pathTest = r pathTeststr
#     try:
#         shutil.rmtree(pathTeststr)
#     except OSError as e:
#         print(e)
#     else:
#         print("The directory is deleted successfully")

basic_parameters = [img_folder_path, save_pred_img_folder_path]
model_parameters = [model_name, backbone_name, num_classes, color_list, ckpt_folder_path]
print("input  Path  {}".format(basic_parameters[0]))
print("output Path  {}".format(basic_parameters[1]))
print(basic_parameters)
print(model_parameters)
print("XXXXX:"+img_folder_path)

# # 模型預測
from time import time
begin = time()
PredFolderImg(basic_parameters, model_parameters,data_height,data_width,color_label, text_label, pos_idx)
#Demo(color_label, text_label, pos_idx).run()  #1110913
print("End of program execution. \n")
end = time()
print(end - begin)

# dirPath = os.getcwd()
# AI_Data_jsonFile=AI_Path+'\B_AI_training\Dentistry_config.json' #modify by  charley 1110830
# # save_model_path = dirPath + "/ckpt"
# print(ckptnmae())
# config["ckptname"] = ckptnmae()  #ckptname best solution
# with open(AI_Data_jsonFile,"w") as f:
#     json.dump(config, f)

#修改為不需刪除 並増加日期年月日時間的資料夾 區隔即不會有重複的狀況 1110831 by charley
#delfolder_compare_output()
#delfolder_compare_labels()









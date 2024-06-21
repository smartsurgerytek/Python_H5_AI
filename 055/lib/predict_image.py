import tensorflow as tf
# from tensorflow import keras
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import segmentation_models as sm
import cv2
import numpy as np
# from keras.utils.data_utils import get_file
# pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.2/efficientnetb7_weights_tf_dim_ordering_tf_kernels.h5"
# weights_path = get_file(pretrained_url.split("/")[-1], pretrained_url)

# import os
from os.path import exists
import os.path as path   # add by charley 1110830
import json              # add by charley 1110830
import operator          # add by charley 1110830
import csv               # add by charley 1110928

import time
import sys

"""
The API for predicting pictures in specific folder.
"""

class PredFolderImg(object):
    """
    Inputs-
    basicParameters: A list that include\n
        1. imgFolderPath: Image folder path which you want to use AI model predicting.
        2. savePredImgFolderPath: The saving folder path of predict image.

    modelParameters: A list that include\n
        1. modelName: Model name which you used to training.
        2. backboneName: Backbone name which you used to training.
        3. numClasses: Number of classes, value needed add 1 (unlabeled).
        4. colorList: A list that each element contains a BGR color list. It color will correspond classes index.
        5. ckptFolderPath: The folder path that checkpoint file(.h5) saving.
    """
    def __init__(self, basicParameters, modelParameters,data_height,data_width,color_dict, text_dict, pos_idx):
        # print("TF version: {}.".format(tf.__version__))
        # print("Keras version: {}.\n".format(keras.__version__))
        print("************************")
        print(basicParameters)
        print("************************")
        
        #'''
        self.started_check_file = "started.csv" #1110928
        self.started_check_flag = 'false'#1110928
        self.color_dict = color_dict
        self.text_dict = text_dict
        self.pos_idx = pos_idx
        self.caries_flag = '' #1111118
        #'''
        # self.color_list = config[color_list]
        # self.correct_label = config[correct_label]
        # self.pos_idx = config[pos_idx]

        self.imgFolderPath = basicParameters[0]
        self.savePredImgFolderPath = basicParameters[1]
        self.ckpname='' #找出最佳的ckpname 預設值為空字串 1110830
        self.cwd = os.getcwd()                                             #1110906 charley
        self.load_folder_path = os.path.join(self.cwd, "load_folder")      #1110906 charley
        self.output_folder_path = os.path.join(self.cwd, "output_folder")  #1110906 charley
        self.dot_count = 0                                                 #1110906 charley

        self.modelName = modelParameters[0]
        self.backboneName = modelParameters[1]
        self.numClasses = modelParameters[2]
        self.colorList = modelParameters[3]
        self.ckptFolderPath = modelParameters[4]
        self.activation = 'sigmoid' if self.numClasses == 1 else 'softmax'
        self.preprocessInput = sm.get_preprocessing(self.backboneName)
        self.data_height=data_height
        self.data_width=data_width
        self.createModel()
        self.run()

    def check_started_file(self):   #1110928
           # 開啟輸出的 CSV 檔案
           with open(self.started_check_file, 'w', newline="") as csvfile:
                 # 建立 CSV 檔寫入器
                writer = csv.writer(csvfile)
                # 寫入一列資料
                writer.writerow(['AI狀態', '高度', '重量'])
                # 寫入另外幾列資料
                writer.writerow(['AI開啟成功', 168, 75]) 
                print("AI開啟成功 ")
                self.started_check_flag = 'true'

    def createModel(self):
        exec("self.model = sm.{}(self.backboneName, classes=self.numClasses, activation=self.activation)".format(self.modelName))

    def loadModelWeight(self, ckptPath):
        self.model.load_weights(ckptPath)

    def createFolder(self, path):  #1110911
        if not os.path.exists(path):
            os.mkdir(path)

    def get_pred_img(self, img_path): #1110911
        self.ori_img = self.cv2_load_img(img_path) # bgr image
        rgb_img = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2RGB)
        self.hight, self.width, _ = rgb_img.shape
        self.pred_label = self.get_pred_label(rgb_img)
        pred_img = np.zeros([512, 512, 3])
        unique_label_list = np.unique(self.pred_label).tolist()
        unique_label_list.remove(0)
        if len(unique_label_list) >= 1:
            for i in unique_label_list:
                idx = np.where(self.pred_label == i)
                pred_img[idx] = self.color_dict[i]
        pred_img = cv2.resize(pred_img, (self.width, self.hight), interpolation=cv2.INTER_NEAREST)
        return pred_img


    def get_pred_label(self, img): #1110911
        resize_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC) 
        resize_img = np.expand_dims(resize_img, axis=0)
        resize_img = tf.convert_to_tensor(resize_img)
        resize_img = tf.cast(resize_img, dtype=tf.float32) / 255.
        pred_logit = self.model.predict(resize_img) # (1, 512, 512, 3)
        pred_label = tf.argmax(pred_logit, axis=-1).numpy() # (1, 512, 512)
        pred_label = np.squeeze(pred_label, axis=0) # (512, 512)
        print('pred_label_這是預測的標籤') #111-11-17
        #print(pred_label) #111-11-17
        return pred_label

    def cv2_load_img(self, path): #1110913
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img

    def cv2_write_img(self, path, img):
        cv2.imencode('.png', img)[1].tofile(path + '.png')

    def warm_up_model(self):  #1110913
        warm_img_path = os.path.join(os.getcwd(), "lib", "warm up", "test.jpg")
        warm_img = self.cv2_load_img(warm_img_path)
        warm_img = cv2.cvtColor(warm_img, cv2.COLOR_BGR2RGB)
        warm_img = warm_img.astype(float)
        self.get_pred_label(warm_img)
        print("Model warm up successfully!")

    def predSingleImg(self, imgPath, savePath): #僅轉單一預測的圖形(tensorflow) 目前停用
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hight, width, _ = img.shape
        resizeImg = cv2.resize(img, (self.data_width, self.data_height), interpolation=cv2.INTER_CUBIC)
        resizeImg = self.preprocessInput(resizeImg)
        resizeImg = np.expand_dims(resizeImg, axis=0)
        predLogit = self.model.predict(resizeImg) # (1, h, w, 3)
        predLabel = tf.argmax(predLogit, axis=-1).numpy() # (1, h, w)
        predLabel = np.squeeze(predLabel, axis=0) # (h, w)
        predImg = np.zeros([self.data_height, self.data_width, 3])

        for i in np.arange(1, len(self.colorList)+1):
            loc = np.where(predLabel == i)
            predImg[loc] = self.colorList[i-1]
            print(self.colorList[i-1])  # 1111117 charley 測試debug使用

        predImg = cv2.resize(predImg, (width, hight), interpolation=cv2.INTER_NEAREST)
        print('savePath_1118 predSingleImg ')  #1118
        print(savePath)    #1111118 charley 測試debug使用
        cv2.imwrite(savePath, predImg)

    def get_mix_img(self, pred_file):
        pred_file = pred_file.astype(np.uint8)
        mix_img = cv2.addWeighted(self.ori_img, 0.6, pred_file, 0.4, 0)
        return mix_img

    def get_label_text_img(self):
        unique_list = np.unique(self.pred_label).tolist()
        unique_list.remove(0)
        
        label_list_len = len(unique_list)
        
        height = (label_list_len // 3) * 30 + 5
        width = 1280
        
        label_img = np.zeros((height, width, 3), np.uint8)
        
        if label_list_len >= 1:
            print(label_list_len)   #1117 表示有找到顏色輸出
            for i in range(label_list_len):
                label_num = unique_list[i]
                print(label_num) #1117 表示有找到顏色輸出的代號 charley
                if (label_num == 2):
                    #print("表示 { } 是 Caries ",label_num)
                    print(' this is Caries no: {0} '.format(label_num)) # 帶數字編號
                    self.caries_flag='true'
                # else:
                #     print(' this is not Caries no: {0} '.format(label_num)) # 帶數字編號
                #     self.caries_flag='false'

                row_idx, col_idx = self.pos_idx[i]
                rectangle_st = (65 + (col_idx-1)*300, 5 + (row_idx-1) * 30)
                rectangle_ed = (135 + (col_idx-1)*300, 30 + (row_idx-1) * 30)
                text_pos = (rectangle_ed[0]+5, rectangle_ed[1]-5)
                print(self.color_dict[label_num]) #1117 表示有找到顏色輸出charley 
                cv2.rectangle(label_img, rectangle_st, rectangle_ed, self.color_dict[label_num], -1)
                cv2.putText(label_img, self.text_dict[label_num], text_pos, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (255,255,255), 1, cv2.LINE_AA)
                #print(label_img) #1117
            
        label_img = self.img_resize(label_img, 2*self.width)  #因為要放兩張圖 
        #label_img = self.img_resize(label_img, self.width)   #1110928
            
        return label_img

        #def get_label_text_img(self):
        #     unique_list = np.unique(self.pred_label).tolist()
        # unique_list.remove(0)
        
        # label_list_len = len(unique_list)
        
        # height = (label_list_len // 3) * 30 + 5
        # width = 1280
        
        # label_img = np.zeros((height, width, 3), np.uint8)
        
        # if label_list_len >= 1:
        #     for i in range(label_list_len):
        #         label_num = unique_list[i]
        #         row_idx, col_idx = self.pos_idx[i]
        #         rectangle_st = (65 + (col_idx-1)*400, 5 + (row_idx-1) * 30)
        #         rectangle_ed = (135 + (col_idx-1)*400, 30 + (row_idx-1) * 30)
        #         text_pos = (rectangle_ed[0]+5, rectangle_ed[1]-5)
                
        #         cv2.rectangle(label_img, rectangle_st, rectangle_ed, self.color_dict[label_num], -1)
        #         cv2.putText(label_img, self.text_dict[label_num], text_pos, cv2.FONT_HERSHEY_COMPLEX_SMALL,
        #                     1, (255,255,255), 1, cv2.LINE_AA)
            
        # # label_img = self.img_resize(label_img, 2*self.width)
        # label_img = self.img_resize(label_img, self.width) #1110928
            
        # return label_img

    def img_resize(self, image, output_img_width):
        height, width = image.shape[0], image.shape[1]
        resize_img = cv2.resize(image, (output_img_width, int(height * output_img_width / width)))
        return resize_img

    def get_compare_img(self, mix_img, text_img):
        comp_img = np.zeros((self.hight+text_img.shape[0], 2*self.width, 3), np.uint8)
        comp_img[:self.hight, 0:self.width] = self.ori_img
        comp_img[:self.hight, self.width:] = mix_img
        comp_img[self.hight:, :] = text_img
        return comp_img
       

    def create_folders(self):
        if not os.path.isdir(self.load_folder_path):
           os.mkdir(self.load_folder_path)
        if not os.path.isdir(self.output_folder_path):
           os.mkdir(self.output_folder_path)
        
    def run_ckpt(self): #Add by charley 1110830 找出最佳的ckpname 
        dic = {}
        for ckptFileName in os.listdir(self.ckptFolderPath):
            # print(ckptFileName.split('_')[3]) # 列示全部的ckpt值
            k = ckptFileName.split('_')[0].strip() # key trim
            dic[k] = ckptFileName.split('_')[1].strip()  # value trim
        print(dic)  # 列示全部的ckpt值    
        json_object = json.dumps(dic, indent=4)
        max_keys = [key for key, value in dic.items() if value == max(dic.values())]
        # print(max_keys[0]) #如為多個找出其中一個即可
        print("best solution ckpt-{} !".format(max_keys[0]))
        self.ckpname=max_keys[0] #存入自訂變數
        #return self.ckpname               

    def run(self):        
        self.create_folders() #開啟資料夾
        self.run_ckpt() #list ckptname   
        print(self.ckptFolderPath)     
        for ckptFileName in os.listdir(self.ckptFolderPath):
            weightIdx = ckptFileName.split('_')[0]+'_'+ckptFileName.split('_')[1]
            if ckptFileName.split('_')[0] != self.ckpname: #找出最佳的 by charley
                # print(ckptFileName.split('_')[2])
                # print("Ignore not add folder ckpt-{} !".format(ckptFileName.split('_')[2].strip()))
                continue
            imgSaveFolderPath = os.path.join(self.savePredImgFolderPath, 'pred-'+weightIdx)
            self.createFolder(imgSaveFolderPath)
            print(imgSaveFolderPath)

            ckptPath = os.path.join(self.ckptFolderPath, ckptFileName)
            print(ckptPath)
            self.loadModelWeight(ckptPath)

            self.warm_up_model() # 1110913

            print(self.imgFolderPath)#Dentistry_1988
            #while XXXXXXX
            #for file in os.listdir(self.imgFolderPath):
              #  print(file)
              #  if (file == 'test_0090.png'):
              #      print("input data {}".format(file))
              #      fileName = os.path.splitext(file)[0] #不要副檔名
              #      imgPath = os.path.join(self.imgFolderPath, file)
					#print("input fileName data {} ".format(fileName))
               #     print(imgSaveFolderPath)
               #     savePath = os.path.join(self.savePredImgFolderPath, fileName + '_pred_0906.png')
               #    print("output data {}".format(savePath) )
               #     self.predSingleImg(imgPath, savePath)

            try:
                 while True:
                    #for file in os.listdir(self.imgFolderPath):
                    if os.listdir(self.load_folder_path) != []:
                        # print("\nfind file {} !".format(os.listdir(self.load_folder_path)))
                        for file_name in os.listdir(self.load_folder_path):
                            save_name = time.strftime('%Y-%m-%d H%H_M%M_S%S', time.localtime())
                            file_path = os.path.join(self.load_folder_path, file_name)
                            write_path = os.path.join(self.output_folder_path, save_name)
                            if os.path.isfile(write_path + '.png'):
                                same_name_count += 1
                                write_path = write_path + '_' + str(same_name_count) 
                                print(write_path)
                            else:
                                same_name_count = 0
                            time.sleep(0.01)

                           

                            
                            print("write_path")                            
                            print("write_path {}".format(write_path))

                            #fileName = os.path.splitext(file_name)[0] #不要副檔名
                            #imgPath = os.path.join(self.imgFolderPath, file_name) #imgPath這個要副檔名
                            imgPath = os.path.join(self.load_folder_path, file_name)
                            print("imgPath {}".format(imgPath))

                            pred_img = self.get_pred_img(imgPath)
                            #pred_img = self.get_pred_img(file_path) #有異常
                            self.caries_flag='false'
                            label_text_img = self.get_label_text_img()

                            print(self.caries_flag)
                            if (self.caries_flag =='true'): #1111118 add by charley
                                write_path = write_path + '_CARIES'
                            elif (self.caries_flag =='false'):
                                write_path = write_path + '_NONE'
                            else:
                                continue
                            
                            mix_img = self.get_mix_img(pred_img)#分析結果
                            comp_img = self.get_compare_img(mix_img, label_text_img)

                            #以下3選一的組合影像寫入***
                            #self.cv2_write_img(write_path, mix_img)          #原圖   1110908
                            #self.cv2_write_img(write_path, label_text_img)   #Color標示名稱 1110908
                            self.cv2_write_img(write_path, comp_img)          #比較圖 1110908---要做修改
                            #****************************************************
                            print("output data write_path {}".format(write_path))
                            #print("input data {}".format(file))
                            #fileName = os.path.splitext(file_name)[0] #不要副檔名
                            #imgPath = os.path.join(self.imgFolderPath, file)
                            #imgPath = os.path.join(self.imgFolderPath, file_name) #imgPath這個要副檔名
					        #print("input fileName data {} ".format(fileName))
                            #print(imgSaveFolderPath)
                            #savePath = os.path.join(self.savePredImgFolderPath, fileName + '_pred.png')
                            #savePath2 = os.path.join(self.output_folder_path, fileName + '_pred.png')
                            #print("output data {}".format(savePath))
                            #print("output data {}".format(savePath2))
                            #self.predSingleImg(imgPath, savePath)#原存檔路徑取消
                            #self.predSingleImg(imgPath, savePath2)#原存檔路徑改為output_folder_path
                            if(os.path.exists(file_path)):
                                os.remove(file_path)  #分析完成刪除檔案

                    sys.stdout.write('\r')#1110928
                    if self.started_check_flag == 'false': #1110928
                       self.check_started_file()  # AI開啟成功 #1110928
                    sys.stdout.write('\r')
                    sys.stdout.write("Wait for new Xray pictures " + "."*self.dot_count + " "*8)
                    self.dot_count = (self.dot_count+1) % 8
                    sys.stdout.flush()
                    time.sleep(0.8)

            except KeyboardInterrupt:
                print("")
                pass


        print("Predict ckpt-{} image ssuccessfully!".format(weightIdx))

    
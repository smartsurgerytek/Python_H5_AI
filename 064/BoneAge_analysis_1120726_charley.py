# python script.py 0,1,2 10
import sys
if len(sys.argv)>1:
    Input1 = sys.argv[0]   # script.py
    Input2 = sys.argv[1]   # 0,1,2 10
    analysis_name=Input2
    print(Input1,Input2)
else:
    analysis_name='D:/SurgeryAnalytics/BoneAgeCode/BoneAge/valid/F160/16y-F-239.png'
    # analysis_name='../0_Data/2_split/valid/F160/16y-F-239.png'
import json
import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary

import numpy as np
import matplotlib.pyplot as plt

# Python 把任意系統的路徑轉換成目前的系統格式（關於 / \ 分隔符的）
import os

from lib.cwdS import Cwd_path

cwd = os.getcwd() 
load_folder_path = os.path.join(cwd, "load_folder")      #1110906 charley
output_folder_path = os.path.join(cwd, "output_folder")  #1110906 charley

# def __init__(self):
#         # print("TF version: {}.".format(tf.__version__))
#         # print("Keras version: {}.\n".format(keras.__version__))
#         print("************************")
#         # print(basicParameters)
#         print("************************")        
#         #'''
#         self.started_check_file = "started.csv" #1110928
#         self.started_check_flag = 'false'#1110928
#         # self.color_dict = color_dict
#         # self.text_dict = text_dict
#         # self.pos_idx = pos_idx
#         # self.caries_flag = '' #1111118
#         #'''
#         # self.color_list = config[color_list]
#         # self.correct_label = config[correct_label]
#         # self.pos_idx = config[pos_idx]

#         # self.imgFolderPath = basicParameters[0]
#         # self.savePredImgFolderPath = basicParameters[1]
#         self.ckpname='' #找出最佳的ckpname 預設值為空字串 1110830
#         self.cwd = os.getcwd()                                             #1110906 charley
#         self.load_folder_path = os.path.join(self.cwd, "load_folder")      #1110906 charley
#         self.output_folder_path = os.path.join(self.cwd, "output_folder")  #1110906 charley
#         self.dot_count = 0                                                 #1110906 charley

#         # self.modelName = modelParameters[0]
#         # self.backboneName = modelParameters[1]
#         # self.numClasses = modelParameters[2]
#         # self.colorList = modelParameters[3]
#         # self.ckptFolderPath = modelParameters[4]
#         # self.activation = 'sigmoid' if self.numClasses == 1 else 'softmax'
#         # self.preprocessInput = sm.get_preprocessing(self.backboneName)
#         # self.data_height=data_height
#         # self.data_width=data_width
#         # self.createModel()
#         # self.run()

# # int dot_count
# # load_folder_path = os.path.join(cwd, "load_folder")      #1110906 charley
# # output_folder_path = os.path.join(cwd, "output_folder")  #1110906 charley



def convert_path(path: str) -> str:
    return path.replace(r'\/'.replace(os.sep, ''), os.sep)

convert_path(analysis_name)
print(convert_path(analysis_name))
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
image_transforms = { 
    'train': transforms.Compose([
        transforms.Resize(size=256),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

def create_folders():
    if not os.path.isdir(load_folder_path):
        os.mkdir(load_folder_path)
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)


def predict(model, test_image_name):
    '''
    Function to predict the class of a single test image
    Parameters
        :param model: Model to test
        :param test_image_name: Test image

    '''
    transform = image_transforms['test']
    test_image = Image.open(test_image_name)
    # plt.imshow(test_image)
    test_image_tensor = transform(test_image)
    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        print(idx_to_class)
        topk, topclass = ps.topk(3, dim=1)
        print(topclass)
        print(topk)
        cls = idx_to_class[topclass.cpu().numpy()[0][0]]
        score = topk.cpu().numpy()[0][0]

        for i in range(1):
            print("Predcition", i+1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])
            print(idx_to_class[topclass.cpu().numpy()[0][i]])
            #這裡的輸出值，會有F110=F女性  M=男性 110  表示11歲0個月  115 表示11歲6個月 0.5*12=6個月
            # F095=女性9歲6個月
            
AI_Data_jsonFile='best_epoch.json'
# idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
# print(idx_to_class)
idx_to_class={0: 'F060', 1: 'F070', 2: 'F075', 3: 'F085', 4: 'F090', 5: 'F095', 6: 'F100', 7: 'F105', 8: 'F110', 9: 'F115', 10: 'F120', 11: 'F125', 12: 'F130', 13: 'F135', 14: 'F140', 15: 'F160'}

# jsonFile = r"f:\SurgeryAnalytics\AI_Cases\Dentistry\1_Linknet_efficientnetb7_0_3\A_Data_preprocessing\Dentistry_config.json"  #讀取剛剛資料分析完輸出的JSION檔

# jsonFile = AI_Data_jsonFile  #讀取剛剛資料分析完輸出的JSION檔
# with open(jsonFile, 'r') as f:
#     config = json.load(f)
# best_epoch=config["best_epoch"]

# best_epoch='3498'
# dataset = 'BoneAge'
# model = torch.load("{}_model_{}.pt".format(dataset, best_epoch))
# model = torch.load(best_epoch)

best_epoch='BoneAge_resnet152_3498_For3500.pth'     #最佳檔案 
model = torch.load(best_epoch)

# analysis_name='../0_Data/2_split/valid/F160/16y-F-239.png'
create_folders() #開啟資料夾
# predict(model, analysis_name)
# Load Data from folders
#computeTestSetAccuracy(model, loss_func)   #
try:
    dot_count = 1 # 將dot_count設定為1 
    # load_folder_path = 'D:\\HomeWork\\AI_PY\\055_pytorch\\055\\load_folder\\'
    while True:
#for file in os.listdir(self.imgFolderPath):
        if os.listdir() != []:
            print("\nfind input file {} !".format(os.listdir(load_folder_path)))
            for file_name in os.listdir(load_folder_path):
                save_name = time.strftime('%Y-%m-%d H%H_M%M_S%S', time.localtime())
                file_path = os.path.join(load_folder_path, file_name)
                print("\n分析檔案{} !".format(file_path))
                predict(model, file_path)
                if os.path.isfile(file_path):
                    print("檔案存在 before {} !".format(file_path))
                else:
                    print("檔案不存在 before {} !".format(file_path))
                time.sleep(15.01)
                write_path = os.path.join(output_folder_path, save_name)
                if os.path.isfile(write_path + '.png'):
                    same_name_count += 1
                    write_path = write_path + '_' + str(same_name_count) 
                    print(write_path)          
                else:
                    same_name_count = 0

                # if os.path.isfile(file_path):
                #     print("\n file check success {} !".format(os.listdir(file_path)))
                # 檢查檔案是否存在
                if os.path.isfile(file_path):
                    print("檔案存在 after {} !".format(file_path))
                else:
                    print("檔案不存在 after {} !".format(file_path))

                try:
                    # print("\n del file success {} !".format(os.listdir(file_path)))
                    if(os.path.exists(file_path)):
                        os.remove(file_path)  #分析完成刪除檔案
                        print("分析完成刪除檔案移除Success")
                except IOError: # 捕捉輸出入錯誤
                    print("分析完成刪除檔案移除值的元素 Error")
                except:
                    print("分析完成刪除檔案移除值的元素 Error")              
                
                

        # sys.stdout.write('\r')
        sys.stdout.write("Wait for new pictures " + "."*dot_count + " "*8)
        dot_count = (dot_count+1) % 8
        sys.stdout.flush()
        time.sleep(2.8)

except KeyboardInterrupt:
    print("")
    pass

          

        
    #     print("write_path")                            
    #     print("write_path {}".format(write_path))

    #     #fileName = os.path.splitext(file_name)[0] #不要副檔名
    #     #imgPath = os.path.join(self.imgFolderPath, file_name) #imgPath這個要副檔名
    #     imgPath = os.path.join(self.load_folder_path, file_name)
    #     print("imgPath {}".format(imgPath))
    #     time.sleep(0.8) #避免存檔不完全11120725
    #     pred_img = self.get_pred_img(imgPath)
    #     #pred_img = self.get_pred_img(file_path) #有異常
    #     self.caries_flag='false'
    #     label_text_img = self.get_label_text_img()

    #     print(self.caries_flag)
    #     if (self.caries_flag =='true'): #1111118 add by charley
    #         write_path = write_path + '_CARIES'
    #     elif (self.caries_flag =='false'):
    #         write_path = write_path + '_NONE'
    #     else:
    #         continue
        
    #     mix_img = self.get_mix_img(pred_img)#分析結果
    #     comp_img = self.get_compare_img(mix_img, label_text_img)

    #     #以下3選一的組合影像寫入***
    #     #self.cv2_write_img(write_path, mix_img)           #原圖   1110908
    #     #self.cv2_write_img(write_path, label_text_img)    #Color標示名稱 1110908
    #     self.cv2_write_img(write_path, comp_img)           #比較圖 1110908---要做修改
    #     #****************************************************
    #     print("output data write_path {}".format(write_path))
    #     #print("input data {}".format(file))
        #   fileName = os.path.splitext(file_name)[0] #不要副檔名
    #     #imgPath = os.path.join(self.imgFolderPath, file)
          #imgPath = os.path.join(self.imgFolderPath, file_name) #imgPath這個要副檔名
        #   print("input fileName data {} ".format(fileName))
    #     #print(imgSaveFolderPath)
    #     #savePath = os.path.join(self.savePredImgFolderPath, fileName + '_pred.png')
    #     #savePath2 = os.path.join(self.output_folder_path, fileName + '_pred.png')
    #     #print("output data {}".format(savePath))
    #     #print("output data {}".format(savePath2))
    #     #self.predSingleImg(imgPath, savePath)#原存檔路徑取消
    #     #self.predSingleImg(imgPath, savePath2)#原存檔路徑改為output_folder_path     

            # sys.stdout.write('\r')#1110928
                # if self.started_check_flag == 'false': #1110928
                #    self.check_started_file()  # AI開啟成功 #1110928
        # sys.stdout.write('\r')
        # sys.stdout.write("Wait for new surgical pictures " + "."*dot_count + " "*8)
        # dot_count = (dot_count+1) % 8
        # sys.stdout.flush()
        # time.sleep(1.8)

# except KeyboardInterrupt:
#     print("")
#     pass
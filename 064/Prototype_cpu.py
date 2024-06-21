# python script.py 0,1,2 10
# 修改更新日期: 1120728 
import sys
if len(sys.argv)>1:
    Input1 = sys.argv[0]   # script.py
    Input2 = sys.argv[1]   # 0,1,2 10
    analysis_name=Input2
    print(Input1,Input2)
else:
    analysis_name='  ' #目前已不使用
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
from pathlib import Path
# Python 把任意系統的路徑轉換成目前的系統格式（關於 / \ 分隔符的）
import os
from lib.cwdS import Cwd_path
import shutil
from PIL import Image, ImageDraw, ImageFont
import csv                             # add by charley 1120728

cwd = os.getcwd() 
# int dot_count = 0
load_folder_path = os.path.join(cwd, "load_folder")       #1120728 charley
output_folder_path = os.path.join(cwd, "output_folder")   #1120728 charley
save_name = 'test'
started_check_file = "started.csv"    #1120728 charley
started_check_flag = 'false'          #1120728 charley

def create_folders():
    if not os.path.isdir(load_folder_path):
        os.mkdir(load_folder_path)
    if not os.path.isdir(output_folder_path):
        os.mkdir(output_folder_path)

def convert_path(path: str) -> str:
    return path.replace(r'\/'.replace(os.sep, ''), os.sep)

# 轉換為中文字 'F145'=='女性14歲5月' 
# by charley
def convert_predict(ss): 
    if int(ss[1:-1])<=6:
        ans= "女性6歲前"
    else:
        ans = "女性"+ss[1:-1]+"歲"+ss[3]+"月"
    return ans

def check_started_file():  #1120728
    # 開啟輸出的 CSV 檔案
    with open(started_check_file, 'w', newline="") as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        # 寫入一列資料
        writer.writerow(['AI狀態', '高度', '重量'])
        # 寫入另外幾列資料
        writer.writerow(['AI開啟成功', 168, 75]) 
        print("AI開啟成功 ")
        return 'true'

# convert_path(analysis_name)
# print(convert_path(analysis_name))

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


def predict( model, test_image_name):
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
    # if torch.cuda.is_available():
    #     test_image_tensor = test_image_tensor.view(1, 3, 224, 224).cuda()
    # else:
    test_image_tensor = test_image_tensor.view(1, 3, 224, 224)
    
    with torch.no_grad():
        model.eval()#1120804
        model.cpu() 
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        print(idx_to_class)
        topk, topclass = ps.topk(3, dim=1)
        print(topclass)
        print(topk)
        cls = idx_to_class[topclass.cpu().numpy()[0][0]]
        score = topk.cpu().numpy()[0][0]
        tmp_score=0.01
        for i in range(3):
            print("Predcition", i+1, ":", idx_to_class[topclass.cpu().numpy()[0][i]], ", Score: ", topk.cpu().numpy()[0][i])
            print(idx_to_class[topclass.cpu().numpy()[0][i]])
            print(topk.cpu().numpy()[0][i])
            score=topk.cpu().numpy()[0][i]
            if score > tmp_score:
                save_name= os.path.splitext(test_image_name)[0]+'_'+idx_to_class[topclass.cpu().numpy()[0][i]]+'.PNG'
                tmp_score=score
            # filename_without_ext = os.path.splitext(test_image_name)[0]
            # filename = Path(path).name
            
        print(save_name)
        #save_name_new = os.path(save_name).name

            #這裡的輸出值，會有F110=F女性  M=男性 110  表示11歲0個月  115 表示11歲6個月 0.5*12=6個月
            # F095=女性9歲6個月
        return save_name
            
AI_Data_jsonFile='best_epoch.json'
# idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}
# print(idx_to_class)
idx_to_class={0: 'F060', 1: 'F070', 2: 'F075', 3: 'F085', 4: 'F090', 5: 'F095', 6: 'F100', 7: 'F105', 8: 'F110', 9: 'F115', 10: 'F120', 11: 'F125', 12: 'F130', 13: 'F135', 14: 'F140', 15: 'F160'}

# jsonFile = r"f:\SurgeryAnalytics\AI_Cases\Dentistry\1_Linknet_efficientnetb7_0_3\A_Data_preprocessing\Dentistry_config.json" 
# 讀取剛剛資料分析完輸出的JSION檔

# jsonFile = AI_Data_jsonFile  #讀取剛剛資料分析完輸出的JSION檔
# with open(jsonFile, 'r') as f:
#     config = json.load(f)
# best_epoch=config["best_epoch"]

# best_epoch='3498'
# dataset = 'BoneAge'
# model = torch.load("{}_model_{}.pt".format(dataset, best_epoch))
# model = torch.load(best_epoch)

best_epoch='BoneAge_resnet152_3498_For3500.pth'  #最佳檔案 
#model = torch.load(best_epoch)

model = torch.load(best_epoch, map_location=torch.device('cpu'))

# analysis_name='../0_Data/2_split/valid/F160/16y-F-239.png'
create_folders() #開啟資料夾
# predict(model, analysis_name)
# Load Data from folders
# computeTestSetAccuracy(model, loss_func) #
try:    
    dot_count = 1 # 將dot_count設定為1 
    while True:
    #for file in os.listdir(self.imgFolderPath):
        if os.listdir() != []:
            # print("\nfind input file {} !".format(os.listdir(load_folder_path)))
            for file_name in os.listdir(load_folder_path):
                # save_name = time.strftime('%Y-%m-%d H%H_M%M_S%S', time.localtime()) #另存以時間命名暫時取消
                save_name = file_name #暫存原始名稱
                file_path = os.path.join(load_folder_path, file_name)
                print("\n分析檔案{} !".format(file_path))
                write_path = os.path.join(output_folder_path, Path(predict(model, file_path)).name)
                print("Output data write_path {}".format(write_path))

                #*************複製檔案至另一個路徑***************Start
                # import shutil
                # # specify the source file path
                # source = "C:/Users/John/Documents/hot.txt"
                # # specify the destination file path and new name
                # destination = "D:/Backup/cool.txt"
                # # copy the file using shutil.copyfile()
                # shutil.copyfile(source, destination)
                # shutil.copyfile(file_path,write_path)
                #*************複製檔案至另一個路徑***************End
          
                
                #*************文字寫入image**********************Start
                picture = Image.open(file_path)
                # # Create a draw object
                draw = ImageDraw.Draw(picture)
                # # Define font size and font type
                font = ImageFont.truetype('simsun.ttc', 20) #simsun.ttc 這個字型才不會亂字
                # # Define text to be written on image
                text =("Original Image :"+(Path(write_path).name).split('_')[0]+"\n"+"Prediction Ans :"+convert_predict((Path(write_path).name).split('_')[1][:-4]))
                # 說明 : convert_predict 
                # convert_predict((Path(write_path).name).split('_')[1][:-4])  #自行轉換F145=='女性14歳5月'
                # # Define text color
                color = 'rgb(168, 171, 5)'
                # # Define text position
                position = (15, 15)
                # # Write text on image
                draw.text(position, text, fill=color, font=font)
                # # Save the image with the specified file name
                picture.save(write_path)
                #*************文字寫入image**********************End

                if os.path.isfile(file_path):
                    print("檔案存在 before Del {} !".format(file_path))
                else:
                    print("檔案不存在 before Del {} !".format(file_path))
                time.sleep(1.01)
                
                if os.path.isfile(write_path + '.png'): #固定附屬檔名為png
                    same_name_count += 1
                    write_path = write_path + '_' + str(same_name_count) 
                    print(write_path)          
                else:
                    same_name_count = 0

                # 檢查檔案是否存在
                if os.path.isfile(file_path):
                    print("檔案存在 after Del {} !".format(file_path))
                else:
                    print("檔案不存在 after Del {} !".format(file_path))

                try:
                    if(os.path.exists(file_path)):
                        os.remove(file_path)  #分析完成刪除檔案
                        print("分析完成刪除檔案移除Success")
                except IOError: # 捕捉輸出入錯誤
                    print("分析完成刪除檔案移除值的元素 Error")
                except:
                    print("分析完成刪除檔案移除值的元素 Error")              
                
        if started_check_flag == 'false': #1110928
            started_check_flag = check_started_file()  # AI開啟成功 #1110928        
        # 跑馬寫在同一列且起始位置為0
        sys.stdout.write('\r')
        sys.stdout.write("Wait for new surgical pictures " + "."*dot_count + " "*18)
        dot_count = (dot_count+1) % 18
        sys.stdout.flush()
        # 跑馬寫在同一列且起始位置為0
        time.sleep(1.8)

except KeyboardInterrupt:
    print("")
    pass  

  

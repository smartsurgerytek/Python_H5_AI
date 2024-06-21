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

predict(model, analysis_name)
# Load Data from folders
#computeTestSetAccuracy(model, loss_func)   #
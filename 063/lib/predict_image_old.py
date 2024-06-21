import tensorflow as tf
from tensorflow import keras
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import segmentation_models as sm
import cv2
import numpy as np

# import os
import os.path as path   # add by charley 1110830
import json              # add by charley 1110830
import operator          # add by charley 1110830 

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
    def __init__(self, basicParameters, modelParameters,data_height,data_width):
        print("TF version: {}.".format(tf.__version__))
        print("Keras version: {}.\n".format(keras.__version__))
        self.imgFolderPath = basicParameters[0]
        self.savePredImgFolderPath = basicParameters[1]
        self.ckpname='' #找出最佳的ckpname 預設值為空字串 1110830

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

    def createModel(self):
        exec("self.model = sm.{}(self.backboneName, classes=self.numClasses, activation=self.activation)".format(self.modelName))

    def loadModelWeight(self, ckptPath):
        self.model.load_weights(ckptPath)

    def createFolder(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def predSingleImg(self, imgPath, savePath):
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

        predImg = cv2.resize(predImg, (width, hight), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(savePath, predImg)
        
        
    def run_ckpt(self): #modify by charley 1110830 找出最佳的ckpname 
        dic = {}
        for ckptFileName in os.listdir(self.ckptFolderPath):
            # print(ckptFileName.split('_')[3]) # 列示全部的ckpt值
            k = ckptFileName.split('_')[2].strip() # key trim
            dic[k] = ckptFileName.split('_')[3].strip()  # value trim
        print(dic)  # 列示全部的ckpt值    
        json_object = json.dumps(dic, indent=4)
        max_keys = [key for key, value in dic.items() if value == max(dic.values())]
        print(max_keys[0]) #如為多個找出其中一個即可
        print("best solution ckpt-{} !".format(max_keys[0]))
        self.ckpname=max_keys[0] #存入自訂變數
        #return self.ckpname               

    def run(self):
        
        self.run_ckpt() #list ckptname        
        for ckptFileName in os.listdir(self.ckptFolderPath):
            weightIdx = ckptFileName.split('_')[0]+'_'+ckptFileName.split('_')[1]+'_'+ckptFileName.split('_')[2]
            if ckptFileName.split('_')[2] != self.ckpname: #找出最佳的 by charley
                #print(ckptFileName.split('_')[2])
                print("Ignore not add folder ckpt-{} !".format(ckptFileName.split('_')[2].strip()))
                continue
            imgSaveFolderPath = os.path.join(self.savePredImgFolderPath, 'pred-'+weightIdx)
            self.createFolder(imgSaveFolderPath)

            ckptPath = os.path.join(self.ckptFolderPath, ckptFileName)
            self.loadModelWeight(ckptPath)

            for file in os.listdir(self.imgFolderPath):
                fileName = os.path.splitext(file)[0]
                imgPath = os.path.join(self.imgFolderPath, file)
                savePath = os.path.join(imgSaveFolderPath, fileName + '_pred.png')
                self.predSingleImg(imgPath, savePath)

            print("Predict ckpt-{} image ssuccessfully!".format(weightIdx))

    
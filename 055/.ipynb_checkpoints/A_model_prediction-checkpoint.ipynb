{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Segmentation Models: using `tf.keras` framework.\n",
      "TF version: 2.1.0.\n",
      "Keras version: 2.2.4-tf.\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "import segmentation_models as sm\n",
    "import os\n",
    "import cv2\n",
    "import numpy as np\n",
    "from lib.predict_image import *\n",
    "ckptname='test'  #最佳CKPT編號 1110830\n",
    "os.environ['CUDA_VISIBLE_DEVICES'] = '0'\n",
    "sm.set_framework('tf.keras')\n",
    "sm.framework()\n",
    "print(\"TF version: {}.\".format(tf.__version__))\n",
    "print(\"Keras version: {}.\".format(keras.__version__))\n",
    "            \n",
    "def solve_cudnn_error():\n",
    "    gpus = tf.config.experimental.list_physical_devices('GPU')\n",
    "    if gpus:\n",
    "        try:\n",
    "            # Currently, memory growth needs to be the same across GPUs\n",
    "            for gpu in gpus:\n",
    "                tf.config.experimental.set_memory_growth(gpu, True)\n",
    "            logical_gpus = tf.config.experimental.list_logical_devices('GPU')\n",
    "            print(len(gpus), \"Physical GPUs,\", len(logical_gpus), \"Logical GPUs\")\n",
    "        except RuntimeError as e:\n",
    "            # Memory growth must be set before GPUs have been initialized\n",
    "            print(e)\n",
    "          \n",
    "           \n",
    "solve_cudnn_error()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "cwd = os.getcwd()\n",
    "# print(cwd)\n",
    "AI_Path = cwd.split('\\C_pred')[0]\n",
    "# print(AI_Path)\n",
    "AI_Data_jsonFile=AI_Path+'\\B_AI_training\\Dentistry_config.json'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "# jsonFile= = r'F:\\SurgeryAnalytics\\AI_Cases\\Dentistry\\1_Linknet_efficientnetb7_0_3\\B_AI_training\\Dentistry_config.json'\n",
    "jsonFile=AI_Data_jsonFile\n",
    "with open(jsonFile,\"r\") as f:\n",
    "    config = json.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'surgeryName': 'Dentistry',\n",
       " 'label_folder_path': 'F:\\\\SurgeryAnalytics\\\\AI_Cases\\\\Dentistry2/0_Data/0_ori',\n",
       " 'trans_folder_path': 'F:\\\\SurgeryAnalytics\\\\AI_Cases\\\\Dentistry2/0_Data/1_trans',\n",
       " 'split_folder_path': 'F:\\\\SurgeryAnalytics\\\\AI_Cases\\\\Dentistry2/0_Data/2_split',\n",
       " 'correct_label': ['Alveolar_bone',\n",
       "  'Caries',\n",
       "  'Crown',\n",
       "  'Dentin',\n",
       "  'Enamel',\n",
       "  'Implant',\n",
       "  'Mandibular_alveolar_nerve',\n",
       "  'Maxillary_sinus',\n",
       "  'Periapical_lesion',\n",
       "  'Post_and_core',\n",
       "  'Pulp',\n",
       "  'Restoration',\n",
       "  'Root_canal_filling'],\n",
       " 'detect_label_list': ['Alveolar_bone',\n",
       "  'Caries',\n",
       "  'Crown',\n",
       "  'Dentin',\n",
       "  'Enamel',\n",
       "  'Implant',\n",
       "  'Mandibular_alveolar_nerve',\n",
       "  'Maxillary_sinus',\n",
       "  'Periapical_lesion',\n",
       "  'Post_and_core',\n",
       "  'Pulp',\n",
       "  'Restoration',\n",
       "  'Root_canal_filling'],\n",
       " 'user_name': '沈易達',\n",
       " 'model': 'Linknet',\n",
       " 'BACKBONE': 'efficientnetb7',\n",
       " 'num_classes': 15,\n",
       " 'save_ckpt_Folder': 'F:\\\\SurgeryAnalytics\\\\AI_Cases\\\\Dentistry2\\\\B_AI_training/ckpt/20220830055055',\n",
       " 'data_height': 384,\n",
       " 'data_width': 384,\n",
       " 'save_pred_img_folder_path': 'F:\\\\SurgeryAnalytics\\\\AI_Cases\\\\Dentistry2\\\\C_pred/prediction/20220831164250',\n",
       " 'color_list': [[0, 240, 255],\n",
       "  [65, 127, 0],\n",
       "  [0, 0, 255],\n",
       "  [113, 41, 29],\n",
       "  [122, 21, 135],\n",
       "  [0, 148, 242],\n",
       "  [4, 84, 234],\n",
       "  [0, 208, 178],\n",
       "  [52, 97, 148],\n",
       "  [121, 121, 121],\n",
       "  [212, 149, 27],\n",
       "  [206, 171, 255],\n",
       "  [110, 28, 216]],\n",
       " 'ckptname': 'n07',\n",
       " 'save_compare_img_folder_path': 'F:\\\\SurgeryAnalytics\\\\AI_Cases\\\\Dentistry2\\\\C_pred/compare_output/20220831164250'}"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "config"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 參數設定\n",
    "## [Model ref](https://github.com/qubvel/segmentation_models)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import time\n",
    "time.localtime()\n",
    "train_time=time.strftime(\"%Y%m%d%H%M%S\", time.localtime())\n",
    "\n",
    "cwd = os.getcwd()\n",
    "DATA_DIR = config[\"split_folder_path\"]# 轉好檔的標記資料路徑(label為mask(即.png格式))\n",
    "x_test_dir = os.path.join(DATA_DIR, 'test', 'img')\n",
    "y_test_dir = os.path.join(DATA_DIR, 'test', 'label')\n",
    "img_folder_path = x_test_dir # 要預測的圖片資料夾路徑\n",
    "\n",
    "save_pred_img_folder_path = r'prediction'+ \"/\"+train_time # 儲存預測圖片的根目錄資料夾路徑\n",
    "if not os.path.isdir(save_pred_img_folder_path):\n",
    "    os.makedirs(save_pred_img_folder_path)\n",
    "    \n",
    "#增加預測對照圖的顯示位置1110831 add by charley\n",
    "save_compare_img_folder_path = r'compare_output'+ \"/\"+train_time # 儲存預測對照圖的根目錄資料夾路徑 \n",
    "if not os.path.isdir(save_compare_img_folder_path):\n",
    "    os.makedirs(save_compare_img_folder_path)\n",
    "#增加預測對照圖的顯示位置1110831 add by charley    \n",
    "    \n",
    "model_name = config[\"model\"] # 使用的模型名稱 (Unet/FPN/Linknet/PSPNet)\n",
    "backbone_name = config[\"BACKBONE\"]# 使用的backbone名稱\n",
    "num_classes = config[\"num_classes\"] # 模型預測的類別數量, 此數值要+1(for unlabel)\n",
    "color_list = [\n",
    "                [0,240,255],#Alveolar_bone\n",
    "                [65,127,0],#'Caries',\n",
    "                [0,0,255],#Crown\n",
    "                [113,41,29],#Dentin\n",
    "                [122,21,135],#Enamel\n",
    "                [0,148,242],#Implant\n",
    "                [4,84,234],#Mandibular_alveolar_nerve\n",
    "                [0,208,178],#Maxillary_sinus\n",
    "                [52,97,148],#Periapical_lesion\n",
    "                [121,121,121],#Post_and_core\n",
    "                [212,149,27],#Pulp\n",
    "                [206,171,255],#Restoration\n",
    "                [110,28,216],#Root_canal_filling\n",
    "                ]\n",
    "                              # 針對資料前處理的correct_label清單, 依序填入想要顯示的BGR色碼 (註:是BGR不是RGB), background顏色不用填.\n",
    "data_height=config[\"data_height\"]\n",
    "data_width=config[\"data_width\"]\n",
    "\n",
    "# MODEL=model_name\n",
    "# if MODEL == 'PSPNet':\n",
    "#     data_height=384\n",
    "#     data_width=384\n",
    "    # tf.compat.v1.disable_eager_execution()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "F:\\SurgeryAnalytics\\AI_Cases\\Dentistry2\\B_AI_training/ckpt/20220830055055\n",
      "compare_output/20220831173819\n"
     ]
    }
   ],
   "source": [
    "ckpt_folder_path = config[\"save_ckpt_Folder\"] # 存放model weight的資料夾路徑\n",
    "# ckpt_folder_path=\"F:\\\\SurgeryAnalytics\\\\AI_Cases\\\\Dentistry/1_Linknet_efficientnetb7_0_1/B_AI_training/ckpt/20220712171222\"\n",
    "print(ckpt_folder_path)\n",
    "config[\"save_pred_img_folder_path\"]  = f\"{cwd}/{save_pred_img_folder_path}\"\n",
    "#save_compare_img_folder_path\n",
    "config[\"save_compare_img_folder_path\"]  = f\"{cwd}/{save_compare_img_folder_path}\"\n",
    "print(save_compare_img_folder_path)   #增加預測對照圖的顯示位置1110831 add by charley\n",
    "config[\"color_list\"] = color_list\n",
    "with open(jsonFile,\"w\") as f:\n",
    "    json.dump(config, f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os.path as path   # add by charley 1110830\n",
    "import json              # add by charley 1110830\n",
    "import operator          # add by charley 1110830\n",
    "def ckptnmae():\n",
    "    #print(ckpt_folder_path)\n",
    "     dic = {}\n",
    "     for ckptFileName in os.listdir(ckpt_folder_path):\n",
    "         #print(ckptFileName.split('_')[3]) # 列示全部的ckpt值\n",
    "         k = ckptFileName.split('_')[2].strip() # key trim\n",
    "         dic[k] = ckptFileName.split('_')[3].strip()  # value trim\n",
    "     print(dic)  # 列示全部的ckpt值    \n",
    "     json_object = json.dumps(dic, indent=4)\n",
    "     max_keys = [key for key, value in dic.items() if value == max(dic.values())]\n",
    "     print(max_keys[0]) #如為多個找出其中一個即可\n",
    "     print(\"best solution ckpt-{} !\".format(max_keys[0]))\n",
    "     ckpname=max_keys[0] #存入自訂變數\n",
    "     return max_keys[0]\n",
    "\n",
    "# AI_Cases/Dentistry2/C_pred/compare_output\n",
    "# AI_Cases/Dentistry2/C_pred/labels\n",
    "import shutil\n",
    "def delfolder_compare_output():\n",
    "    pathTeststr = AI_Path +\"/C_pred/compare_output\"\n",
    "    print(pathTeststr)\n",
    "    #pathTest = r pathTeststr\n",
    "    #os.remove(path)\n",
    "    try:\n",
    "        shutil.rmtree(pathTeststr)\n",
    "    except OSError as e:\n",
    "        print(e)\n",
    "    else:\n",
    "        print(\"The directory is deleted successfully\")\n",
    "    \n",
    "def delfolder_compare_labels():\n",
    "    pathTeststr = AI_Path +\"/C_pred/labels\"\n",
    "    print(pathTeststr)\n",
    "    #pathTest = r pathTeststr\n",
    "    try:\n",
    "        shutil.rmtree(pathTeststr)\n",
    "    except OSError as e:\n",
    "        print(e)\n",
    "    else:\n",
    "        print(\"The directory is deleted successfully\")\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['F:\\\\SurgeryAnalytics\\\\AI_Cases\\\\Dentistry2/0_Data/2_split\\\\test\\\\img', 'prediction/20220831173819']\n",
      "['Linknet', 'efficientnetb7', 15, [[0, 240, 255], [65, 127, 0], [0, 0, 255], [113, 41, 29], [122, 21, 135], [0, 148, 242], [4, 84, 234], [0, 208, 178], [52, 97, 148], [121, 121, 121], [212, 149, 27], [206, 171, 255], [110, 28, 216]], 'F:\\\\SurgeryAnalytics\\\\AI_Cases\\\\Dentistry2\\\\B_AI_training/ckpt/20220830055055']\n"
     ]
    }
   ],
   "source": [
    "basic_parameters = [img_folder_path, save_pred_img_folder_path]\n",
    "model_parameters = [model_name, backbone_name, num_classes, color_list, ckpt_folder_path]\n",
    "print(basic_parameters)\n",
    "print(model_parameters)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 模型預測"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "TF version: 2.1.0.\n",
      "Keras version: 2.2.4-tf.\n",
      "\n",
      "WARNING:tensorflow:Large dropout rate: 0.5125 (>0.5). In TensorFlow 2.x, dropout() uses dropout rate instead of keep_prob. Please ensure that this is intended.\n",
      "WARNING:tensorflow:Large dropout rate: 0.525 (>0.5). In TensorFlow 2.x, dropout() uses dropout rate instead of keep_prob. Please ensure that this is intended.\n",
      "WARNING:tensorflow:Large dropout rate: 0.5375 (>0.5). In TensorFlow 2.x, dropout() uses dropout rate instead of keep_prob. Please ensure that this is intended.\n",
      "WARNING:tensorflow:Large dropout rate: 0.55 (>0.5). In TensorFlow 2.x, dropout() uses dropout rate instead of keep_prob. Please ensure that this is intended.\n",
      "WARNING:tensorflow:Large dropout rate: 0.5625 (>0.5). In TensorFlow 2.x, dropout() uses dropout rate instead of keep_prob. Please ensure that this is intended.\n",
      "{'n01': '0.749', 'n02': '0.737', 'n03': '0.776', 'n04': '0.793', 'n06': '0.802', 'n07': '0.810'}\n",
      "n07\n",
      "best solution ckpt-n07 !\n",
      "Ignore not add folder ckpt-n01 !\n",
      "Ignore not add folder ckpt-n02 !\n",
      "Ignore not add folder ckpt-n03 !\n",
      "Ignore not add folder ckpt-n04 !\n",
      "Ignore not add folder ckpt-n06 !\n",
      "Predict ckpt-Linknet_efficientnetb7_n07 image ssuccessfully!\n",
      "80.22022461891174\n"
     ]
    }
   ],
   "source": [
    "from time import time\n",
    "begin = time()\n",
    "PredFolderImg(basic_parameters, model_parameters,data_height,data_width)\n",
    "end = time()\n",
    "print(end - begin)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'n01': '0.749', 'n02': '0.737', 'n03': '0.776', 'n04': '0.793', 'n06': '0.802', 'n07': '0.810'}\n",
      "n07\n",
      "best solution ckpt-n07 !\n",
      "n07\n",
      "{'n01': '0.749', 'n02': '0.737', 'n03': '0.776', 'n04': '0.793', 'n06': '0.802', 'n07': '0.810'}\n",
      "n07\n",
      "best solution ckpt-n07 !\n"
     ]
    }
   ],
   "source": [
    "dirPath = os.getcwd()\n",
    "AI_Data_jsonFile=AI_Path+'\\B_AI_training\\Dentistry_config.json' #modify by  charley 1110830\n",
    "# save_model_path = dirPath + \"/ckpt\"\n",
    "print(ckptnmae())\n",
    "config[\"ckptname\"] = ckptnmae()  #ckptname best solution\n",
    "with open(AI_Data_jsonFile,\"w\") as f:\n",
    "    json.dump(config, f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'surgeryName': 'Dentistry',\n",
       " 'label_folder_path': 'F:\\\\SurgeryAnalytics\\\\AI_Cases\\\\Dentistry2/0_Data/0_ori',\n",
       " 'trans_folder_path': 'F:\\\\SurgeryAnalytics\\\\AI_Cases\\\\Dentistry2/0_Data/1_trans',\n",
       " 'split_folder_path': 'F:\\\\SurgeryAnalytics\\\\AI_Cases\\\\Dentistry2/0_Data/2_split',\n",
       " 'correct_label': ['Alveolar_bone',\n",
       "  'Caries',\n",
       "  'Crown',\n",
       "  'Dentin',\n",
       "  'Enamel',\n",
       "  'Implant',\n",
       "  'Mandibular_alveolar_nerve',\n",
       "  'Maxillary_sinus',\n",
       "  'Periapical_lesion',\n",
       "  'Post_and_core',\n",
       "  'Pulp',\n",
       "  'Restoration',\n",
       "  'Root_canal_filling'],\n",
       " 'detect_label_list': ['Alveolar_bone',\n",
       "  'Caries',\n",
       "  'Crown',\n",
       "  'Dentin',\n",
       "  'Enamel',\n",
       "  'Implant',\n",
       "  'Mandibular_alveolar_nerve',\n",
       "  'Maxillary_sinus',\n",
       "  'Periapical_lesion',\n",
       "  'Post_and_core',\n",
       "  'Pulp',\n",
       "  'Restoration',\n",
       "  'Root_canal_filling'],\n",
       " 'user_name': '沈易達',\n",
       " 'model': 'Linknet',\n",
       " 'BACKBONE': 'efficientnetb7',\n",
       " 'num_classes': 15,\n",
       " 'save_ckpt_Folder': 'F:\\\\SurgeryAnalytics\\\\AI_Cases\\\\Dentistry2\\\\B_AI_training/ckpt/20220830055055',\n",
       " 'data_height': 384,\n",
       " 'data_width': 384,\n",
       " 'save_pred_img_folder_path': 'F:\\\\SurgeryAnalytics\\\\AI_Cases\\\\Dentistry2\\\\C_pred/prediction/20220831173819',\n",
       " 'color_list': [[0, 240, 255],\n",
       "  [65, 127, 0],\n",
       "  [0, 0, 255],\n",
       "  [113, 41, 29],\n",
       "  [122, 21, 135],\n",
       "  [0, 148, 242],\n",
       "  [4, 84, 234],\n",
       "  [0, 208, 178],\n",
       "  [52, 97, 148],\n",
       "  [121, 121, 121],\n",
       "  [212, 149, 27],\n",
       "  [206, 171, 255],\n",
       "  [110, 28, 216]],\n",
       " 'ckptname': 'n07',\n",
       " 'save_compare_img_folder_path': 'F:\\\\SurgeryAnalytics\\\\AI_Cases\\\\Dentistry2\\\\C_pred/compare_output/20220831173819'}"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "config"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "#修改為不需刪除 並増加日期年月日時間的資料夾 區隔即不會有重複的狀況 1110831 by charley\n",
    "#delfolder_compare_output()\n",
    "#delfolder_compare_labels()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "AI",
   "language": "python",
   "name": "ai"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

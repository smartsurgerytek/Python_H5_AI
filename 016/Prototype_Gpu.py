import cv2
import numpy as np
import time
import sys
import os
import csv                 #1110928
import datetime            #1130304
from datetime import date  #1130304
      

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras
from lib.models.EfficientNet import EfficientNet
from PIL import Image
import segmentation_models as sm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

sm.set_framework('tf.keras')
sm.framework()


config = tf.compat.v1.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction=0.7
config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 0.6 sometimes works better for folks
# session  = tf.compat.v1.Session(config=config)

def solve_cudnn_error():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            # config = tf.compat.v1.ConfigProto()
            # config.gpu_options.allow_growth = True
            # sess = tf.compat.v1.Session(config=config)

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

class Demo():

    def __init__(self, color_dict, text_dict, pos_idx):
        # Setting initial parameters
        self.started_check_file = "started.csv" #1110928
        self.started_check_flag = 'false'#1110928
        self.color_dict = color_dict
        self.text_dict = text_dict
        self.pos_idx = pos_idx
        self.model = EfficientNet(num_classes=11)
        self.model.build(input_shape=(None, 512, 512, 3))
        print("Create model successfully!")
        self.cwd = os.getcwd()
        self.load_folder_path = os.path.join(self.cwd, "load_folder")
        self.output_folder_path = os.path.join(self.cwd, "output_folder")
        self.dot_count = 0
        self.output = "counter_log_" + "0305" + ".csv"
        #self.output = "counter_log_"+ str(date.day)+".csv"
        self.select_out=2    #1 標識圖  #2 標識圖+label #3 比較圖+label

    def check_started_file(self):   #1110928
           # 開啟輸出的 CSV 檔案
           with open(self.started_check_file, 'w', newline="") as csvfile:
                 # 建立 CSV 檔寫入器
                writer = csv.writer(csvfile)
                # 寫入一列資料
                writer.writerow(['姓名', '身高', '體重'])
                # 寫入另外幾列資料
                writer.writerow(['AI開啟成功', 168, 75]) 
                print("AI開啟成功 ")
                self.started_check_flag = 'true'

    def check_transfer_file_time(self): #1110928
           # 開啟輸出的 CSV 檔案
        from datetime import date  #1130304
        output = self.output #"counter_mean.csv"
        # 開啟輸出的 CSV 檔案
        #digit=1
        #for digit in range(1,11,1):
            # if digit == 10:
            #     break
            #     print()
        # self.t1 = datetime.datetime.now()
        # for i in range(10000000):
        #     pass
        self.t2 = datetime.datetime.now()
        t = self.t2-self.t1
        print(str(date.today()) + "##" + str(self.t1) + "##" + str(self.t2) + "##" + str(round(t.total_seconds()*1000))+"ms")        
        with open(output, 'a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([date.today(), self.t1, self.t2, round(t.total_seconds()*1000),' ms'])
        #print("CSV--Python時間分析存檔成功 ")
        self.t2=""
        self.t1=""
 
    def create_folders(self):
        if not os.path.isdir(self.load_folder_path):
            os.mkdir(self.load_folder_path)
        if not os.path.isdir(self.output_folder_path):
            os.mkdir(self.output_folder_path)

    def load_model_weight(self):
        weight_path = os.path.join(self.cwd, "lib", "weight", "ckpt", "EfficientNet_ckpt")
        self.model.load_weights(weight_path)
        print("Load weight successfully!")

    def cv2_load_img(self, path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        return img

    def cv2_write_img(self, path, img):
        cv2.imencode('.png', img)[1].tofile(path + '.png')

    def warm_up_model(self):
        warm_img_path = os.path.join(os.getcwd(), "lib", "warm up", "test.jpg")
        warm_img = self.cv2_load_img(warm_img_path)
        warm_img = cv2.cvtColor(warm_img, cv2.COLOR_BGR2RGB)
        warm_img = warm_img.astype(float)
        self.get_pred_label(warm_img)
        print("Model warm up successfully!")

    def get_pred_img(self, img_path):
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

    def get_pred_label(self, img):



        resize_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC) 
        resize_img = np.expand_dims(resize_img, axis=0)
        resize_img = tf.convert_to_tensor(resize_img)
        resize_img = tf.cast(resize_img, dtype=tf.float32) / 255.

        pred_logit = self.model.predict(resize_img) # (1, 512, 512, 3)
        pred_label = tf.argmax(pred_logit, axis=-1).numpy() # (1, 512, 512)
        pred_label = np.squeeze(pred_label, axis=0) # (512, 512)
        return pred_label

    def get_mix_img(self, pred_file):
        pred_file = pred_file.astype(np.uint8)
        mix_img = cv2.addWeighted(self.ori_img, 0.6, pred_file, 0.4, 0)
        return mix_img

    def get_label_text_img_2(self):
        unique_list = np.unique(self.pred_label).tolist()
        unique_list.remove(0)
        
        label_list_len = len(unique_list)
        
        height = (label_list_len // 3) * 30 + 5
        width = 1280
        
        label_img = np.zeros((height, width, 3), np.uint8)
        
        if label_list_len >= 1:
            for i in range(label_list_len):
                label_num = unique_list[i]
                row_idx, col_idx = self.pos_idx[i]
                rectangle_st = (25 + (col_idx-1)*330, 5 + (row_idx-1) * 30)
                rectangle_ed = (90 + (col_idx-1)*330, 30 + (row_idx-1) * 30)
                text_pos = (rectangle_ed[0]+5, rectangle_ed[1]-5)
                
                cv2.rectangle(label_img, rectangle_st, rectangle_ed, self.color_dict[label_num], -1)
                cv2.putText(label_img, self.text_dict[label_num], text_pos, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (255,255,255), 1, cv2.LINE_AA)
            
        label_img = self.img_resize(label_img, self.width)
            
        return label_img

    def get_label_text_img_3(self):
        unique_list = np.unique(self.pred_label).tolist()
        unique_list.remove(0)
        
        label_list_len = len(unique_list)
        
        height = (label_list_len // 3) * 30 + 5
        width = 1280
        
        label_img = np.zeros((height, width, 3), np.uint8)
        
        if label_list_len >= 1:
            for i in range(label_list_len):
                label_num = unique_list[i]
                row_idx, col_idx = self.pos_idx[i]
                rectangle_st = (25 + (col_idx-1)*330, 5 + (row_idx-1) * 30)
                rectangle_ed = (90 + (col_idx-1)*330, 30 + (row_idx-1) * 30)
                text_pos = (rectangle_ed[0]+5, rectangle_ed[1]-5)
                
                cv2.rectangle(label_img, rectangle_st, rectangle_ed, self.color_dict[label_num], -1)
                cv2.putText(label_img, self.text_dict[label_num], text_pos, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            1, (255,255,255), 1, cv2.LINE_AA)
            
        label_img = self.img_resize(label_img, 2*self.width)
            
        return label_img


    def img_resize(self, image, output_img_width):
        height, width = image.shape[0], image.shape[1]
        resize_img = cv2.resize(image, (output_img_width, int(height * output_img_width / width)))
        return resize_img

    def get_compare_img_3(self, mix_img, text_img):#2張左右各一
        comp_img = np.zeros((self.hight+text_img.shape[0], 2*self.width, 3), np.uint8)
        comp_img[:self.hight, 0:self.width] = self.ori_img
        comp_img[:self.hight, self.width:] = mix_img
        comp_img[self.hight:, :] = text_img
        return comp_img

   
    def get_compare_img_2(self, mix_img, text_img):#1張上下各一
        comp_img = np.zeros((self.hight+text_img.shape[0], self.width, 3), np.uint8)
        comp_img[:self.hight, 0:self.width] = mix_img
        #comp_img[self.hight:, :] = mix_img
        comp_img[self.hight:, :] = text_img
        return comp_img


    def run(self):
        self.create_folders()
        self.load_model_weight()
        self.warm_up_model()
        same_name_count = 0



    # logf = open("download.log", "w")
    # for download in download_list:
    #     try:
    #         # code to process download here
    #     except Exception as e:     # most generic exception you can catch
    #         logf.write("Failed to download {0}: {1}\n".format(str(download), str(e)))
    #         # optional: delete local version of failed download
    #     finally:
    #         # optional clean up code
    #         pass

        logf = open("process_error.log", "a")#1130313 test
        try:
            while True:
                try:
                    # code to process download here                    
                    if os.listdir(self.load_folder_path) != []:
                        print("\nfind file {} !".format(os.listdir(self.load_folder_path)))
                        for file_name in os.listdir(self.load_folder_path):
                            self.t1 = datetime.datetime.now()
                            save_name = time.strftime('%Y-%m-%d H%H_M%M_S%S', time.localtime())
                            file_path = os.path.join(self.load_folder_path, file_name)
                            write_path = os.path.join(self.output_folder_path, save_name)
                            if os.path.isfile(write_path + '.png'):
                                same_name_count += 1
                                write_path = write_path + '_' + str(same_name_count)
                            else:
                                same_name_count = 0
                            time.sleep(0.2) #11305
                            pred_img = self.get_pred_img(file_path)
                            mix_img = self.get_mix_img(pred_img)#分析結果  
                            self.cv2_write_img(write_path, mix_img)#標註圖
                            os.remove(file_path)

                            # if self.select_out==1:                            
                            #    self.cv2_write_img(write_path, mix_img)#標註圖
                            # #    os.remove(file_path)
                            # elif self.select_out==2: 
                            #    label_text_img = self.get_label_text_img_2()                        
                            #    comp_img = self.get_compare_img_2(mix_img, label_text_img)
                            #    self.cv2_write_img(write_path, comp_img)#標註圖+label
                            # #    os.remove(file_path)
                            # else:
                            #    label_text_img = self.get_label_text_img_3()                        
                            #    comp_img = self.get_compare_img_3(mix_img, label_text_img)
                            #    self.cv2_write_img(write_path, pred_img)#比較圖(左+右)+label
                            #    os.remove(file_path)
                            
                            self.check_transfer_file_time()
                            # show_img = Image.open(write_path + '.png')
                            # show_img.show()
                    else:
                        sys.stdout.write('\r')#1110928
                        if self.started_check_flag == 'false': #1110928
                            self.check_started_file()  # AI開啟成功 #1110928
                        sys.stdout.write('\r')
                        sys.stdout.write("Wait for new surgical pictures " + "."*self.dot_count + " "*4)                   
                        self.dot_count = (self.dot_count+1) % 4
                        sys.stdout.flush()
                        time.sleep(0.2)

                except Exception as e:     # most generic exception you can catch
                        logf.write("Failed to process {0}: {1}\n".format(str(time.localtime()), str(e))) 
                # optional: delete local version of failed download
                # finally:
                # # optional clean up code
                #     pass

                    #**************************************************
                    # if os.listdir(self.load_folder_path) != []:
                    #     print("\nfind file {} !".format(os.listdir(self.load_folder_path)))
                    #     for file_name in os.listdir(self.load_folder_path):
                    #         self.t1 = datetime.datetime.now()
                    #         save_name = time.strftime('%Y-%m-%d H%H_M%M_S%S', time.localtime())
                    #         file_path = os.path.join(self.load_folder_path, file_name)
                    #         write_path = os.path.join(self.output_folder_path, save_name)
                    #         if os.path.isfile(write_path + '.png'):
                    #             same_name_count += 1
                    #             write_path = write_path + '_' + str(same_name_count)
                    #         else:
                    #             same_name_count = 0
                    #         time.sleep(0.2) #11305
                    #         pred_img = self.get_pred_img(file_path)
                    #         mix_img = self.get_mix_img(pred_img)#分析結果  
                    #         self.cv2_write_img(write_path, mix_img)#標註圖
                    #         os.remove(file_path)

                    #         # if self.select_out==1:                            
                    #         #    self.cv2_write_img(write_path, mix_img)#標註圖
                    #         # #    os.remove(file_path)
                    #         # elif self.select_out==2: 
                    #         #    label_text_img = self.get_label_text_img_2()                        
                    #         #    comp_img = self.get_compare_img_2(mix_img, label_text_img)
                    #         #    self.cv2_write_img(write_path, comp_img)#標註圖+label
                    #         # #    os.remove(file_path)
                    #         # else:
                    #         #    label_text_img = self.get_label_text_img_3()                        
                    #         #    comp_img = self.get_compare_img_3(mix_img, label_text_img)
                    #         #    self.cv2_write_img(write_path, pred_img)#比較圖(左+右)+label
                    #         #    os.remove(file_path)
                            
                    #         self.check_transfer_file_time()
                    #         # show_img = Image.open(write_path + '.png')
                    #         # show_img.show()
                    # else:

                    #     sys.stdout.write('\r')#1110928
                    #     if self.started_check_flag == 'false': #1110928
                    #     self.check_started_file()  # AI開啟成功 #1110928
                    #     sys.stdout.write('\r')
                    #     sys.stdout.write("Wait for new surgical pictures " + "."*self.dot_count + " "*4)                   
                    #     self.dot_count = (self.dot_count+1) % 4
                    #     sys.stdout.flush()
                    #     time.sleep(0.2) 

        except Exception as e:     # most generic exception you can catch
            logf.write("Failed to process {0}: {1}\n".format(str(time.localtime()), str(e))) #1130313

        except KeyboardInterrupt:
            print("")
            logf.write("KeyboardInterrup {0}: {1}\n".format(str(time.localtime()), str(e))) #1130313
            pass
        finally:
            # # optional clean up code
            pass

if __name__ == "__main__":
    # B, G, R
    color_label = {
        1:[158, 95, 229], # Artery
        2:[4, 84, 234], # Ureter
        3:[12, 160, 244],  # External iliac artery
        4:[143, 50, 81],  # Fallopian Tube
        5:[148, 57, 0],  # Ovary
        6:[126, 187, 100],  # Round ligament
        7:[184, 123, 118],  # IP ligament
        8:[221, 193, 85],  # Peritoneum
        9:[55, 172, 46],  # Scalpel
        10:[27, 90, 167]  # Injury wound
    }

    text_label = {
        1:"Artery",
        2:"Ureter",
        3:"External iliac artery",
        4:"Fallopian Tube",
        5:"Ovary",
        6:"Round ligament",
        7:"IP ligament",
        8:"Peritoneum",
        9:"Scalpel",
        10:"Injury wound"
    }

    pos_idx = [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4), (3,1), (3,2)]#顯示標註的起位置
    solve_cudnn_error()
    Demo(color_label, text_label, pos_idx).run()
    print("End of program execution. \n")
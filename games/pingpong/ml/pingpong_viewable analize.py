
import pickle
import os 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm

from mpl_toolkits.mplot3d import Axes3D
'''
fig = plt.figure()
axis = fig.gca(projection='3d')
'''
fig = plt.figure()
fig.add_subplot(111, projection='3d')

item_index = 0
load_file = 0
folder_path = "C:\\Users\\bigse\\MLGame\\games\\pingpong\\log"     #指定LOG資料夾位置
folder_content = os.listdir(folder_path)
hit =[]
x=[]
y=[]
Z=[]
ball_start=[]
ball_speed=[]
Frame = [ ]
Status = [ ]
Ballposition = [ ]
Platformposition = [ ]
Bricks = [ ]
Hard_bricks = [ ]
kp = [ ]
Bricks_sum = [0,0] 

for item in folder_content:
        if os.path.isdir(folder_path + '\\' + item):
                print('資料夾：' + item)
        elif os.path.isfile(folder_path + '\\' + item):
               file_name = str( folder_path + '\\' + item )
               game_file = open (file_name, "rb")
               data_list = pickle.load(game_file)
               data_list = data_list["ml_1P"]["scene_info"] #將資料夾中檔案的必要場警資訊全部匯入data List
               game_file.close()
               """
               try:
                   os.remove(file_name)
               except OSError as e:
                   print(e)
               game_file.close()
               """
        else:
                print('無法辨識：' + item)
#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
                
        j=0
        plt.plot(x,y)
        hit.append(98)

        for i in range (0,len(data_list)):
            #x.append(data_list[i]['ball'][0])
            #y.append(data_list[i]['ball'][1])
            
            '''
            if data_list[i]['ball'][1]>420-5-1 and data_list[i]['ball_speed'][1]<0:
                ball_start.append(data_list[i]['ball'])
                ball_speed.append(data_list[i]['ball_speed'])
                x.append(data_list[i]['ball'][0])
                y.append(data_list[i]['ball_speed'][0])
            if data_list[i]['ball'][1]<80+5+1 and data_list[i]['ball_speed'][1]>0:
                ball_start.append(data_list[i]['ball'])
                ball_speed.append(data_list[i]['ball_speed'])
                Z.append(data_list[i]['ball'][0]/180)
            '''
            if data_list[i]['ball'][1]>420-5-1 :
                ball_start.append(data_list[i]['ball'])
                ball_speed.append(data_list[i]['ball_speed'])
                hit.append(data_list[i]['ball'][0])
                x.append(hit[-2])
                y.append(data_list[i]['ball_speed'][0])
                #Z.append(int(hit[-2])-int(hit[-1]))
                Z.append(int(hit[-1]))
        print(j)
        
        #plt.scatter(x, y, c=Z, cmap=plt.cm.Paired, edgecolors='k')
        
        ax122 = plt.subplot(1,2,2, projection='3d')
        ax122.scatter(x, y, Z, c='r')
        plt.title("retune point", fontsize=12)
        plt.xlabel("hit point", fontsize=12)
        plt.ylabel("speed x", fontsize=12)
        plt.show()
 

print('load '+ str(item_index) + ' files')

import pickle
import os 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
'''
fig = plt.figure()
axis = fig.gca(projection='3d')
'''

item_index = 0
load_file = 0
folder_path = "C:\\Users\\bigse\\MLGame\\games\\pingpong\\log"     #指定LOG資料夾位置
folder_content = os.listdir(folder_path)
hit_right = []
hit_left = []
x=[]
y=[]
Z=[]
ball_start_right=[]
ball_start_left=[]
ball_stop_right = []
ball_stop_left = []
ball_speed_right=[]
ball_speed_left=[]
Frame = [ ]
Status = [ ]
Ballposition = [ ]
Platform_move = [ ]
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
        item_index = item_index + 1   
        j=0
        hit_right.append(98)
        hit_left.append(98)

        for i in range (0,len(data_list)):

            if data_list[i]['ball'][1]>420-5-1 and data_list[i]['ball_speed'][0] > 0 and data_list[i]['status'] == "GAME_ALIVE":
                hit_right.append(data_list[i]['ball'][0])
                ball_start_right.append(hit_right[-2])
                ball_speed_right.append(data_list[i]['ball_speed'][0])
                ball_stop_right.append(hit_right[-1])
            elif data_list[i]['ball'][1]>420-5-1 and data_list[i]['ball_speed'][0] < 0 and data_list[i]['status'] == "GAME_ALIVE":
                hit_left.append(data_list[i]['ball'][0])
                ball_start_left.append(hit_left[-2])
                ball_speed_left.append(data_list[i]['ball_speed'][0])
                ball_stop_left.append(hit_left[-1])
                
x1 = np.array(ball_start_right)
x2= np.array(ball_speed_right)
#x = np.hstack((x1[:, np.newaxis], x2[:, np.newaxis]))
x = x1[:, np.newaxis]
y1 = np.array(ball_stop_right)
y = y1[:, np.newaxis]
print('load '+ str(item_index) + ' files')
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=999)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train,y_train)
ylm_bef_scaler = lm.predict(x_test)
from sklearn.metrics import r2_score
R2 = r2_score(y_test,ylm_bef_scaler)
print('R2_right = ',R2)

plt.scatter(x_train, y_train, color='black')
plt.plot(x_train, lm.predict(x_train),color='blue',linewidth=1)
plt.show()
filename = "C:\\Users\\bigse\\MLGame\\games\\pingpong\\LR_example_1P_right.sav"
pickle.dump(lm , open(filename , 'wb'))
#----------------------------------------------------------------------------------------------------------------------------
x1 = np.array(ball_start_left)
x2= np.array(ball_speed_left)
#x = np.hstack((x1[:, np.newaxis], x2[:, np.newaxis]))
x = x1[:, np.newaxis]
y1 = np.array(ball_stop_left)
y = y1[:, np.newaxis]
print('load '+ str(item_index) + ' files')
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=999)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train,y_train)
ylm_bef_scaler = lm.predict(x_test)
from sklearn.metrics import r2_score
R2 = r2_score(y_test,ylm_bef_scaler)
print('R2_left = ',R2)

plt.scatter(x_train, y_train, color='black')
plt.plot(x_train, lm.predict(x_train),color='blue',linewidth=1)
plt.show()
filename = "C:\\Users\\bigse\\MLGame\\games\\pingpong\\LR_example_1P_left.sav"
pickle.dump(lm , open(filename , 'wb'))
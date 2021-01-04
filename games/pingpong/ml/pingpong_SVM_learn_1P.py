
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
'''
fig = plt.figure()
fig.add_subplot(111, projection='3d')
'''
item_index = 0
load_file = 0
folder_path = "C:\\Users\\bigse\\MLGame\\games\\pingpong\\log"     #指定LOG資料夾位置
folder_content = os.listdir(folder_path)
hit = []
x=[]
y=[]
Z=[]
ball_start=[]
ball_speed=[]
ball_next=[]
ball_return=[]
Platform_move =[]
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
               load_file = load_file + 1
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
        #plt.plot(x,y)
        hit.append(98)

        for i in range (0,len(data_list)-1):
            if data_list[i]['ball'][1]>420-5-1 and data_list[i]['status'] == "GAME_ALIVE":

                hit.append(data_list[i]['ball'][0])
                ball_start.append(hit[-2])
                ball_speed.append(data_list[i]['ball_speed'][0])
                if hit[-1] < 65:
                    ball_return.append(35)
                elif hit[-1] > 155:
                    ball_return.append(165)
                else:
                    ball_return.append(100)
 

print('load '+ str(item_index) + ' files')

#----------------------------------------------------------------------------------------------------------------------------

Ballarray = np.array(Ballposition[:-1])
x1 = np.array(ball_start)
x2= np.array(ball_speed)
x = np.hstack((x1[:, np.newaxis], x2[:, np.newaxis]))
y1 = np.array(ball_return)
y = y1[:, np.newaxis]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=999)

#----------------------------------------------------------------------------------------------------------------------------
from sklearn.svm import SVR

svr = SVR(gamma=0.001,C = 1,epsilon = 0.1,kernel = 'rbf')

svr.fit(x_train,y_train)

ysvr_bef_scaler = svr.predict(x_test)

from sklearn.metrics import r2_score

R2 = r2_score(y_test,ysvr_bef_scaler)
print('R2 = ',R2)
#----------------------------------------------------------------------------------------------------------------------------
'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_stdnorm = scaler.transform(x_train)
knn.fit(x_train_stdnorm ,y_train)
x_test_stdnorm = scaler.transform(x_test)
yknn_aft_scaler = knn.predict(x_test_stdnorm)
acc_knn_aft_scaler = accuracy_score(yknn_aft_scaler,y_test)
'''

#----------------------------------------------------------------------------------------------------------------------------
filename = "C:\\Users\\bigse\\MLGame\\games\\pingpong\\SVM_example_1P.sav"
pickle.dump(svr , open(filename , 'wb'))
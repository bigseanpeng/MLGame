#from sklearn import svm
import pickle
import os 
import numpy as np
import matplotlib.pyplot as plt
item_index = 0
load_file = 0
folder_path = "C:\\Users\\bigse\\MLGame\\games\\arkanoid\\arkanoid"     #指定LOG資料夾位置
folder_content = os.listdir(folder_path)
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
               data_list = data_list["ml"]["scene_info"] #將資料夾中檔案的必要場警資訊全部匯入data List
               if len(data_list[-1]['bricks']) == 0: #如果最後一幕場景中方塊都被消除，該場景才被拿來學習
                   load_file = load_file + 1
                   game_file.close()
               else:
                   game_file.close()
                   try:
                       os.remove(file_name)
                   except OSError as e:
                       print(e)
                   else:
                       print("File is deleted successfully") #如果最後一幕場景中還有方塊未被消除，刪除該LOG檔，剃除無效資料
               game_file.close()
        else:
                print('無法辨識：' + item)

        if len(data_list[-1]['bricks']) == 0:
                item_index = item_index + 1
                for i in range (0,len(data_list)):
                    Frame.append(data_list[i]['frame'])
                    Status.append(data_list[i]['status'])
                    Ballposition.append(data_list[i]['ball'])
                    Platformposition.append(data_list[i]['platform'])
                    for j in range (0,len(data_list[i]['bricks'])): #將每一場景方塊的X軸與Y軸分別加總起來
                        Bricks_sum [0] = Bricks_sum [0] + (data_list[i]['bricks'][j][0])
                        Bricks_sum [1] = Bricks_sum [1] + (data_list[i]['bricks'][j][1])
                    for j in range (0,len(data_list[i]['hard_bricks'])):#硬方塊也一起記錄加總起來
                        Bricks_sum [0] = Bricks_sum [0] + (data_list[i]['hard_bricks'][j][0])
                        Bricks_sum [1] = Bricks_sum [1] + (data_list[i]['hard_bricks'][j][1])
                    Bricks_sum [0] = Bricks_sum [0] * (len(data_list[i]['bricks'])+len(data_list[i]['hard_bricks']))#將每一場景方塊X軸的加總乘以方塊數量來避免特定場景數值重複
                    Bricks.append(Bricks_sum)
                    Bricks_sum = [0,0]
        Brickistributed = np.array(Bricks)[:-1]
        #print("Brickistributed")
        #print(Brickistributed)
        data_list = []
print('load '+ str(item_index) + ' files')


#----------------------------------------------------------------------------------------------------------------------------
PlatX = np.array(Platformposition)[:,0][:,np.newaxis]
PlatX_next = PlatX[1:,:]
instruct = (PlatX_next - PlatX[0:len(PlatX_next),0][:, np.newaxis])/5

BallX = np.array(Ballposition)[:,0][:,np.newaxis]
BallX_next = BallX[1:,:]
vx = (BallX_next - BallX[0:len(BallX_next),0][:,np.newaxis])

BallY = np.array(Ballposition)[:,1][:,np.newaxis]
BallY_next = BallY[1:,:]
vy = (BallY_next - BallY[0:len(BallY_next),0][:,np.newaxis])

Ballarray = np.array(Ballposition[:-1])
x = np.hstack((Ballarray , PlatX[0:-1,0][:,np.newaxis],vx,vy))

y = instruct

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
filename = "C:\\Users\\bigse\\MLGame\\games\\arkanoid\\SVM_example.sav"
pickle.dump(svr , open(filename , 'wb'))
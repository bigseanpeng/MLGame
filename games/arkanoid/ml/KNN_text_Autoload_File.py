import pickle
import os
game_file =[]
data_list = []
folder_path = "D:\\MLGame_8_0\\MLGame-master\\games\\arkanoid\\log"
folder_content = os.listdir(folder_path)
for item in folder_content:
        if os.path.isdir(folder_path + '\\' + item):
                print('資料夾：' + item)
        elif os.path.isfile(folder_path + '\\' + item):
               print('"'+ item + '"')
               file_name = str(folder_path + '\\' + item)
               with open ('"' + folder_path + '\\' + item + '"' , "rb") as game_file[item]:
                       data_list[item] = pickle.load(game_file[item])
                       data_list[item] =data_list[item]["ml"]["scene_info"]
        else:
                print('無法辨識：' + item)
Frame = [ ]
Status = [ ]
Ballposition = [ ]
Platformposition = [ ]
Bricks = [ ]
for item in folder_content:
        for i in range (0,len(data_list[item])):
                Frame.append(data_list[item][i]['frame'])
                Status.append(data_list[item][i]['status'])
                Ballposition.append(data_list[item][i]['ball'])
                Platformposition.append(data_list[item][i]['platform'])
                Bricks.append(data_list[item][i]['bricks'])
#----------------------------------------------------------------------------------------------------------------------------
import numpy as np
PlatX = np.array(Platformposition)[:,0][:,np.newaxis]
PlatX_next = PlatX[1:,:]
instruct = (PlatX_next - PlatX[0:len(PlatX_next),0][:, np.newaxis])/5

BallX=np.array(Ballposition)[:,0][:,np.newaxis]
BallX_next=BallX[1:,:]
vx=(BallX_next-BallX[0:len(BallX_next),0][:,np.newaxis])

BallY=np.array(Ballposition)[:,1][:,np.newaxis]
BallY_next=BallY[1:,:]
vy=(BallY_next-BallY[0:len(BallY_next),0][:,np.newaxis])

Ballarray = np.array(Ballposition[:-1])

x = np.hstack((Ballarray , PlatX[0:-1,0][:,np.newaxis],vx,vy))
#x = np.hstack((Ballarray , PlatX[0:-1,0][:,np.newaxis]))
y = instruct

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=41)
#----------------------------------------------------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(x_train,y_train)

yknn_bef_scaler = knn.predict(x_test)
acc_knn_bef_scaler = accuracy_score(yknn_bef_scaler, y_test)
#----------------------------------------------------------------------------------------------------------------------------
filename = "D:\\MLGame_8_0\\MLGame-master\\knn_text2.sav"
pickle.dump(knn , open(filename , 'wb'))

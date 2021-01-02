
import pickle
import os 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import svm
xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))
np.random.seed(0)
X = np.random.rand(500, 2)
#Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
Y = (X.sum(axis=1)*1.5)
Y = np.ceil(Y)-2
# fit the model


item_index = 0
load_file = 0
folder_path = "C:\\Users\\bigse\\MLGame\\games\\pingpong\\log"     #指定LOG資料夾位置
folder_content = os.listdir(folder_path)
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
        for i in range (0,len(data_list)):
            #x.append(data_list[i]['ball'][0])
            #y.append(data_list[i]['ball'][1])
            
            
            if data_list[i]['ball'][1]>420-5-1 and data_list[i]['ball_speed'][1]<0:
                ball_start.append(data_list[i]['ball'])
                ball_speed.append(data_list[i]['ball_speed'])
                x.append(data_list[i]['ball'][0])
                y.append(data_list[i]['ball_speed'][0])
            if data_list[i]['ball'][1]<80+5+1 and data_list[i]['ball_speed'][1]>0:
                ball_start.append(data_list[i]['ball'])
                ball_speed.append(data_list[i]['ball_speed'])
                Z.append(data_list[i]['ball'][0]/180)

                
        print(j)
        
        plt.scatter(x, y, c=Z, cmap=plt.cm.Paired, edgecolors='k')
        plt.show()
        
'''            
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
'''
print('load '+ str(item_index) + ' files')
#print("Brickistributed")
#print(Brickistributed)
#print(data_list)

#----------------------------------------------------------------------------------------------------------------------------
"""
PlatX = np.array(Platformposition)[:,0]
PlatX_next = np.array(PlatX[1:])
PlatX = np.array(PlatX[0:-1])
instruct = (PlatX_next - PlatX)/5

BallX=np.array(Ballposition)[:,0][:,np.newaxis]
BallX_next=BallX[1:,:]
vx=(BallX_next-BallX[0:len(BallX_next),0][:,np.newaxis])

BallY=np.array(Ballposition)[:,1][:,np.newaxis]
BallY_next=BallY[1:,:]
vy=(BallY_next-BallY[0:len(BallY_next),0][:,np.newaxis])

Ballarray = np.array(Ballposition[:-1])


X = Ballarray   #學習的X軸資訊分別為球的位置、板子的位置、球的移動方向以及場景磚塊特徵

PlatDef = np.array(PlatX[:])
PlatDef = PlatDef - np.array(X[:,0])
Ball_Vector = np.hstack([vx,vy])

 

X = X * Ball_Vector
X = X / [4200,6000]
X = X + [0.5,0.5]

Y = (BallX_next[:,0]-100)/5


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=41)

#----------------------------------------------------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(x_train,y_train)

yknn_bef_scaler = knn.predict(x_test)
acc_knn_bef_scaler = accuracy_score(yknn_bef_scaler, y_test)


filename = "C:\\Users\\bigse\\MLGame\\games\\arkanoid\\arkanoid\\knn_text.sav"
pickle.dump(knn , open(filename , 'wb'))
"""
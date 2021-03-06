"""
The template of the script for the machine learning process in game pingpong
python MLGame.py -i ml_play_SVM.py -i ml_play_rule.py pingpong EASY
"""

ball_position_history=[]
import random
class MLPlay:
    def __init__(self, side):
        """
        Constructor
        @param side A string "1P" or "2P" indicates that the `MLPlay` is used by
               which side.
        """
        self.ball_served = False
        self.side = side

    def update(self, scene_info):
        """
        Generate the command according to the received scene information
        """
        global ball_destination_1P
        global ball_destination_2P
        import pickle
        import numpy as np
        filename = "C:\\Users\\bigse\\MLGame\\games\\pingpong\\SVM_example.sav"
        model = pickle.load(open(filename,'rb'))
        hit_deep = 10
        if scene_info["frame"] == 0:
            ball_destination_1P = 98
        if scene_info["status"] != "GAME_ALIVE":
            print(scene_info["ball_speed"])
            return "RESET"

        if not self.ball_served:
            self.ball_served = True
            command = random.choice(["SERVE_TO_LEFT","SERVE_TO_RIGHT"])
            return command
        
        ball_position_history.append(scene_info["ball"])#Get ball location
        platform1_edge_x = scene_info["platform_1P"][0]+35#Get platform1 location
        platform2_edge_x = scene_info["platform_2P"][0]+35#Get platform2 location
#============================================1P control====================================================
        if self.side == "1P" and scene_info['ball'][1]>420-5-1 :
            inp_temp = np.array([scene_info["ball"][0], scene_info["ball_speed"][0]])
            input = inp_temp[np.newaxis, :]
            ball_destination_1P = (model.predict(input)*20)+scene_info["ball"][0]
            print (ball_destination_1P)
        if self.side == "1P" and scene_info["ball_speed"][1]>0:
            ball_destination_1P = scene_info["ball"][0]+ (((420-scene_info["ball"][1])/scene_info["ball_speed"][1])*scene_info["ball_speed"][0])
            while ball_destination_1P < 0 or ball_destination_1P > 195:
                if ball_destination_1P < 0:
                    ball_destination_1P = -ball_destination_1P
                if ball_destination_1P > 195:
                    ball_destination_1P = 195-(ball_destination_1P-195)
            if ball_destination_1P < scene_info["platform_1P"][0]+hit_deep:
                command = "MOVE_LEFT"
            elif ball_destination_1P > platform1_edge_x-hit_deep:
                command = "MOVE_RIGHT"
            else:
                command = "NONE"
            return command

        elif self.side == "1P" and scene_info["ball_speed"][1]<0:
            if ball_destination_1P < scene_info["platform_1P"][0]+hit_deep:
                command = "MOVE_LEFT"
            elif ball_destination_1P > platform1_edge_x-hit_deep:
                command = "MOVE_RIGHT"
            else:
                command = "NONE"
            return command
            
#============================================1P control====================================================

#============================================2P control====================================================
        if self.side == "2P" and scene_info["ball_speed"][1]<0:
            ball_destination_2P = scene_info["ball"][0]+ (((80-scene_info["ball"][1])/scene_info["ball_speed"][1])*scene_info["ball_speed"][0])
            while ball_destination_2P < 0 or ball_destination_2P > 195:
                if ball_destination_2P < 0:
                    ball_destination_2P = -ball_destination_2P
                if ball_destination_2P > 195:
                    ball_destination_2P = 195-(ball_destination_2P-195)
            if ball_destination_2P < scene_info["platform_2P"][0]+hit_deep:
                command = "MOVE_LEFT"
            elif ball_destination_2P > platform1_edge_x-hit_deep:
                command = "MOVE_RIGHT"
            else:
                command = "NONE"
            return command
        elif self.side == "2P" and scene_info["ball_speed"][1]>0:
            ball_destination_2P = 98
            if ball_destination_2P < scene_info["platform_2P"][0]+hit_deep:
                command = "MOVE_LEFT"
            elif ball_destination_2P > platform2_edge_x-hit_deep:
                command = "MOVE_RIGHT"
            else:
                command = "NONE"
            return command
#============================================2P control====================================================
             
    def reset(self):
        """
        Reset the status
        """
        self.ball_served = False



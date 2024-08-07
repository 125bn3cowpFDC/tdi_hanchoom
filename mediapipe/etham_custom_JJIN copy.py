import cv2
import mediapipe as mp
import numpy as np
import json
from collections import OrderedDict

start_frame = 1553
end_frame = start_frame + 101
# 1955
num = '006'


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

file_data = OrderedDict()
file_data["data"] = []
tmp = []
pose_list = []
cnt = 1
a = []

label = 'walking_sawi'
label_index = 1
output_json_name = './data_hanchoom/new_h_train/jump_sawi_' + num + '.json'
cap = cv2.VideoCapture("/home/smaipcjjh/Documents/project/ai_project/STGCN/mediapipe/AIproject_DB/연습_발사위1_kor.mp4")
temp_frame = start_frame

# CAP SETTING
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
 
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if (not ret) or (temp_frame == end_frame):
          print('FINISH (AREA)')
          break

        if ret is True:
        # Recolor image to RGB
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          image.flags.writeable = False
      
        # Make detection
          results = pose.process(image)
    
        # Recolor back to BGR
          image.flags.writeable = True
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks 0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28
        # Joint index:
          # 0 = {0,  "Nose"} -> 0
          # (11+12)/2 = {1,  "Neck"} -> 17,
          # 11 = {2,  "RShoulder"} -> 5,
          # 13 = {3,  "RElbow"} -> 7,
          # 15 = {4,  "RWrist"} -> 9,
          # 12 = {5,  "LShoulder"} -> 6,
          # 14 = {6,  "LElbow"} -> 8,
          # 16 = {7,  "LWrist"} -> 10,
          # 23 = {8,  "RHip"} -> 11,
          # 25 = {9,  "RKnee"} -> 13,
          # 27 = {10, "RAnkle"} -> 15,
          # 24 = {11, "LHip"} -> 12,
          # 26 = {12, "LKnee"} -> 14,
          # 28 = {13, "LAnkle"} -> 16,
          # 2 = {14, "REye"} -> 1,
          # 5 = {15, "LEye"} -> 2,
          # 7 = {16, "REar"} -> 3,
          # 8 = {17, "LEar"} -> 4,

          try:
            print('*** NOW_FRAME: ', temp_frame)
            landmarks = results.pose_landmarks.landmark

            for i in range(0, 29):
              print('CHECK i for save skeleton = ', i)

              if ((i==0) or (i==2) or (i==5) or (i==7) or (i==8) or (i==11) or (i==12) or (i==13) or (i==14) 
                    or (i==15) or (i==16) or (i==23) or (i==24) or (i==25) or (i==26) or (i==27) or (i==28)):
                a.append(round(landmarks[i].x, 3))
                a.append(round(landmarks[i].y, 3))

            a.append(round((landmarks[11].x + landmarks[12].x)/2, 3))
            a.append(round((landmarks[11].y + landmarks[12].y)/2, 3))

            frame_c = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print('* pose count: ', len(a)/2)
            file_data["data"].append({'frame_index': round(cnt) ,"skeleton":[{'pose':a,'score':[1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]}]})
            a=[]
             
          except:
            print('!!!!!!! ERROR !!!!!!!')
            pass
        
        # Render detections
          mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               

         #print(type(landmarks))
         #print("apaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",c)
          cnt += 1
          temp_frame += 1
        cv2.imshow('Mediapipe Feeder', image)

        if cv2.waitKey(10) & 0xFF == ord('q') or ret == False:
            break
    
    
    file_data["label"] = label
    file_data["label_index"] = label_index

    # 저장
    with open(output_json_name, 'w', encoding="utf-8") as make_file:
      json.dump(file_data, make_file, ensure_ascii=False)    
    
    #print(a)
    cap.release()
    cv2.destroyAllWindows()
    print('CREATE ANNOTATION FILE COPLETE!!!')
    #print(json.dumps(file_data, ensure_ascii=False, indent="\t"))
    
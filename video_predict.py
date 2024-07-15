import mediapipe as mp
import cv2
import numpy as np

import torch
import torch.nn as nn

import argparse
import yaml
from torch.autograd import Variable

def import_class(name):
    components = name.split('.')
    print(components)
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    print(mod)
    return mod


def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='./config/st_gcn/hanchoom/predict.yaml',
        help='path to the configuration file')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')

    return parser


def point_return(landmark):

    a = []

    a.append(round(landmark[0].x, 3))
    a.append(round(landmark[0].y, 3))

    a.append(round(landmark[2].x, 3))
    a.append(round(landmark[2].y, 3))

    a.append(round(landmark[5].x, 3))
    a.append(round(landmark[5].y, 3))

    a.append(round(landmark[7].x, 3))
    a.append(round(landmark[7].y, 3))

    a.append(round(landmark[8].x, 3))
    a.append(round(landmark[8].y, 3))

    a.append(round(landmark[11].x, 3))
    a.append(round(landmark[11].y, 3))

    a.append(round(landmark[12].x, 3))
    a.append(round(landmark[12].y, 3))

    a.append(round(landmark[13].x, 3))
    a.append(round(landmark[13].y, 3))

    a.append(round(landmark[14].x, 3))
    a.append(round(landmark[14].y, 3))

    a.append(round(landmark[15].x, 3))
    a.append(round(landmark[15].y, 3))

    a.append(round(landmark[16].x, 3))
    a.append(round(landmark[16].y, 3))

    a.append(round(landmark[23].x, 3))
    a.append(round(landmark[23].y, 3))

    a.append(round(landmark[24].x, 3))
    a.append(round(landmark[24].y, 3))

    a.append(round(landmark[25].x, 3))
    a.append(round(landmark[25].y, 3))

    a.append(round(landmark[26].x, 3))
    a.append(round(landmark[26].y, 3))

    a.append(round(landmark[27].x, 3))
    a.append(round(landmark[27].y, 3))

    a.append(round(landmark[28].x, 3))
    a.append(round(landmark[28].y, 3))

    a.append(round((landmark[11].x + landmark[12].x)/2, 3))
    a.append(round((landmark[11].y + landmark[12].y)/2, 3))

    return a



if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        print("THIS IS: ", p.config)
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()

    #from st_gcn.net import st
    Model = import_class(arg.model)
    model = Model(**arg.model_args)
    model.load_state_dict(torch.load(arg.weights))
    model = model.cuda(0)
    model.eval()

    # -----------------------------------------------------------------------------------------------

    label_path = './8class_labels.txt'
    labeling = {}
    f = open(label_path, 'r')

    while True:
        line = f.readline()
        if not line: break
    
        strings = line.split(' ')
        labeling[int(strings[0])] = strings[1][:-1]
    f.close()


    # -----------------------------------------------------------------------------------------------

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    datapath = "./output_TEST_02.mp4"
    cap = cv2.VideoCapture(datapath)

    start_frame = 0

    # CAP SETTING
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #out = cv2.VideoWriter('output_TEST.mp4', fourcc, fps, (frame_width, frame_height))

    cnt = 0
    landmark_list = []
    data_numpy = np.zeros((3, 100, 18, 1))
    data_numpy_in = np.zeros((3, 100, 18, 1))
    score = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    show_count = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print('CANT OPEN VIDEO')
                break

            else:
                # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                
                # Make detection
                results = pose.process(image)
                
                # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                # Get landmarks
                print('*** NOW_FRAME: ', cnt)
                landmarks = results.pose_landmarks.landmark

                landmark_list = point_return(landmarks)
                print(len(landmark_list))
                if cnt < 100:
                    #print("aaa",data_numpy[0, cnt, :, 0], cnt)
                    data_numpy[0, cnt, :, 0] = landmark_list[0::2] #x좌표
                    data_numpy[1, cnt, :, 0] = landmark_list[1::2] #y좌표
                    data_numpy[2, cnt, :, 0] = score 
                    #print(len(data_numpy[0]))
                elif cnt >= 100:
                    for i in range(99):
                        #print("포",i)
                        data_numpy[:, i, :, 0] = data_numpy[:, i+1, :, 0]
                    data_numpy[0, 99, :, 0] = landmark_list[0::2]
                    data_numpy[1, 99, :, 0] = landmark_list[1::2]
                    data_numpy[2, 99, :, 0] = score
                    
                    if cnt % 10 == 0:
                        print('여기')
                        data_numpy_p = data_numpy.tolist()
                        data_numpy_p = np.array(data_numpy_p)
                        print(data_numpy_p.shape)

                        data_numpy_p[0:2] = data_numpy_p[0:2] - 0.5

                        print(data_numpy_p.shape)
                        data_numpy_p = np.reshape(data_numpy_p, (1, 3, 100, 18, 1))
                        torch.no_grad()
                        data_torch = torch.Tensor(data_numpy_p).float().cuda(0)                 
                        print("들어가는거:\n")
                        print(data_torch.shape)
                        with torch.no_grad():
                            outputs = model(data_torch)

                        _, predicted = torch.max(outputs, 1)

                        softmax_out = torch.nn.functional.softmax(outputs, dim=1)

                        print('OUTPUT: ', outputs)
                        print('SOFTMAX out: ', softmax_out)
                        print('predict: ', predicted)
                        show_count += 1

                    
                    label_name = labeling[int(predicted[0])]
                    this_out = float(softmax_out[0, int(predicted[0])]) * 100
                    cv2.putText(image, label_name +':  ' + str(this_out), (700, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


                    # -----------------------------------------------------------------------------------------------

                    breathing = float(softmax_out[0, 0]) * 100
                    walking_sawi = float(softmax_out[0, 1]) * 100
                    jump_sawi = float(softmax_out[0, 2]) * 100
                    turn_sawi = float(softmax_out[0, 3]) * 100
                    normp_sawi = float(softmax_out[0, 4]) * 100
                    normp_sawi2 = float(softmax_out[0, 5]) * 100
                    everyone_sawi = float(softmax_out[0, 6]) * 100
                    wind_sawi = float(softmax_out[0, 7]) * 100

                    breathing = round(breathing,2)
                    walking_sawi = round(walking_sawi,2)
                    jump_sawi = round(jump_sawi,2)
                    turn_sawi = round(turn_sawi,2)
                    normp_sawi = round(normp_sawi,2)
                    normp_sawi2 = round(normp_sawi2,2)
                    everyone_sawi = round(everyone_sawi,2)
                    wind_sawi = round(wind_sawi,2)

                    cv2.putText(image, 'breathing: ' + str(breathing), (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'walking_sawi: ' + str(walking_sawi), (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'jump_sawi: ' + str(jump_sawi), (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'turn_sawi: ' + str(turn_sawi), (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'normp_sawi: ' + str(normp_sawi), (10, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'normp_sawi2: ' + str(normp_sawi2), (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'everyone_sawi: ' + str(everyone_sawi), (10, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, 'wind_sawi: ' + str(wind_sawi), (10, 330),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                cv2.putText(image, 'predict conunt: ' + str(show_count), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, 'pos frame: ' + str(cnt), (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)



                # Processing

                    
                # Render detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))               

                cnt += 1
            cv2.imshow('ChungChun', image)

            if cv2.waitKey(10) & 0xFF == ord('q') or ret == False:
                break 
            
        #print(a)
        cap.release()
        cv2.destroyAllWindows()
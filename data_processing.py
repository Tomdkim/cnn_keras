import numpy as np
import cv2
import pandas as pd
import glob
from scipy import ndimage

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
from keras.utils import np_utils

import sys

# Data_processing():
# 1. Extract video frames from a video from t_start to t_end for each t_lapse.
# 2. Extract velocities from a velocity csv from t_start to t_end for each t_lapse.
# 3. Prepare input of size_x * size_y * channel & its label (corresponding velocity).
def main(argv):

    # argv: data_processing.py - t_lapse - t_start - t_end - size_x - size_y (channel)
    if (len(argv) != 6):
        print("Wrong number of arguments.")
        exit(1)
    # if (argv[0] % 0.1 != 0):
    #     print("Time lapse must be a multiple of 0.1.")
    #     exit(1)
    input_path_video='/Users/tomkim/Dropbox/deep_learned_ACC_project/data/03072017/20170301_ 20170301_Recorder_008_test_001.avi'
    output_path_video_frames='/Users/tomkim/Desktop/Video_frames'
    input_path_velocities='/Users/tomkim/Dropbox/deep_learned_ACC_project/data/data8.csv'
    output_path_velocities='/Users/tomkim/Desktop/Video_frames/output_velocities.csv'
    t_lapse = float(argv[1])
    t_start = float(argv[2])
    t_end = float(argv[3])
    size_x = int(argv[4])
    size_y = int(argv[5])
    # Extract video frames from a given video input.
    names = extract_video_frames(t_start, t_end, input_path_video, output_path_video_frames, t_lapse)
    # Extract velocities from a given video input.
    extract_velocities(t_start, t_end, input_path_velocities, output_path_velocities, t_lapse, names)

    x, y = process_input(output_path_video_frames, output_path_velocities, size_x, size_y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state=42)
    X_train = np.array([X_train])[0]
    X_test = np.array([X_test])[0]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # filepath = "/home/ubuntu/CNN/weights/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    # callbacks_list = [checkpoint]

    model = cnn_model()
    model.fit(X_train, y_train, nb_epoch = 20, shuffle = 10, batch_size=1, validation_split=0.3, callbacks=callbacks_list)
    return

def cnn_model():

    model = Sequential()
    model.add(Convolution2D(5, 24, 2, border_mode='valid', input_shape=(100, 200, 5), activation='relu'))
    model.add(Convolution2D(4, 36, 2, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
    model.add(Convolution2D(3, 48, 2, activation='relu', border_mode='same'))
    model.add(Convolution2D(2, 64, 2, activation='relu', border_mode='same'))
    model.add(Convolution2D(2, 64, 2, activation='relu', border_mode='same'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(75, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(output_dim=1,activation='relu'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model

def extract_video_frames(t_start, t_end, input_path, output_path, t):

    cap = cv2.VideoCapture(input_path, 0)

    fps = int(np.round(cap.get(cv2.CAP_PROP_FPS))) #get the Frames Per Second of a video
    if fps != 30:
        print('not a 30 FPS Video')

    names = []
    k = t * fps
    
    f_start = t_start * fps
    f_end = t_end * fps + k

    #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter(output_path,fourcc, 5.0, (640,480)) 
    #(name,fourcc-code,fps,frame_size)
    number_of_frame = 0
    while (cap.isOpened()):
        number_of_frame += 1
        ret, frame = cap.read()
        if np.all(ret == True and number_of_frame >= f_start and number_of_frame <= f_end):
            #out.write(frame)
            #cv2.imshow('frame',frame)
            if (number_of_frame % k) == 0:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Convert to black and white
                cv2.imwrite(output_path + '/frame_' + str(number_of_frame) + '.jpg',gray_frame)
                names.append('frame_' + str(number_of_frame))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if number_of_frame > f_end:
            break

    # Release everything if job is finished
    cap.release()
    #out.release()
    cv2.destroyAllWindows()

    for i in range(1,10):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    return names

def extract_velocities(t_start, t_end, input_path, output_path, t, names):
    interval = t / 0.01 #account for the fact that there are 100 data_points per second
    velocities = np.transpose(np.genfromtxt(input_path, delimiter=',',)[:,1][t_start * 100 : t_end * 100 + 1][0::interval]) # + 1?
    names_array = np.transpose(np.asarray(names))
    frame = pd.DataFrame({ 'frame_name' : names_array, 'velocity': velocities})
    frame.to_csv(output_path, header=True, index=False, index_label=False)

    # v=t*100 #account for the fact that there are 100 data_points per second
    # velocities = np.transpose(np.genfromtxt(input_path, delimiter=',',)[:,1][t_start*100:t_end*100][0::v])
    # names_array= np.transpose(np.asarray(names))
    # frame=pd.DataFrame({ 'frame_name' : names_array, 'velocity': velocities})
    # frame.to_csv(output_path,header=True,index=False,index_label=False)

def process_input(path_f, path_v, size_x, size_y):
    img_validation = pd.read_csv(path_v) # csv file : Name of the framne, Corresponding Speed
    img_validation.columns = ['index', 'speed']
    img_validation['index'] = img_validation['index'].apply(lambda x: str(x) + '.jpg', 1)

    image_dict = []
    for filename in glob.glob(path_f + '/*.jpg'):
        img = cv2.imread(filename)
        img = cv2.resize(img, (size_y, size_x))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_dict.append({'index': filename.split('/')[-1], 'raw': img}) # extract only names of imgs and append them to the dictionary

    tmp = pd.DataFrame.from_dict(image_dict)
    # tmp.head() # 0 - 1479699964422.png (index) - [[[155, 155, 155], [145, 145, 145], [144, 144,... (raw)
    tmp2 = img_validation.merge(tmp)
    # tmp2.head() # 0 - 1479699965366.png (index) - 0.000050 (speed) - [[[155, 155, 155], [154, 154, 154], [153, 153,... (raw)

    x = tmp2['raw'] # X = a column of raw (gray scale or rgb of imgs)
    y = tmp2['speed'].values # Y = a column of corresponding speed

    frame_input = []
    speed_label = []
    # for i in range(1,len(x)):
    #     if i%5==0:
    #         frame_input.append(np.concatenate((np.expand_dims(x[i],2),np.expand_dims(x[i-1],2),np.expand_dims(x[i-2],2),np.expand_dims(x[i-3],2),np.expand_dims(x[i-4],2)),axis=2))
    #         speed_label.append((y[i]+y[i-1]+y[i-2]+y[i-3]+y[i-4])/5)

    for i in range(0, len(x) - 9):
        frame_input.append(np.concatenate((np.expand_dims(x[i], 2), np.expand_dims(x[i + 1], 2), np.expand_dims(x[i + 2], 2), np.expand_dims(x[i + 3], 2), np.expand_dims(x[i + 4], 2)), axis = 2))
        speed_label.append(y[i + 9]) # y[i+4] + time_lapse * 5

    x_s = pd.Series(frame_input)
    y_s = np.asarray(speed_label)

    return x_s, y_s

if __name__ == "__main__":
    main(sys.argv)

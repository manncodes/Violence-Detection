import cv2
import os
import numpy as np
import keras
from keras.applications import VGG16
from keras import backend as K
from keras.models import Model
import sys
import time
import multiprocessing
from termcolor import colored
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model = keras.models.load_model('model/vlstm_92.h5')
image_model = VGG16(include_top=True, weights='imagenet')  
model.summary()  
#We will use the output of the layer prior to the final
# classification-layer which is named fc2. This is a fully-connected (or dense) layer.
transfer_layer = image_model.get_layer('fc2')
image_model_transfer = Model(inputs=image_model.input,outputs=transfer_layer.output)
transfer_values_size = K.int_shape(transfer_layer.output)[1]

# Frame size  
img_size = 224

img_size_touple = (img_size, img_size)

# Number of channels (RGB)
num_channels = 3

# Flat frame size
img_size_flat = img_size * img_size * num_channels

# Number of classes for classification (Violence-No Violence)
num_classes = 2

# Number of files to train
_num_files_train = 1

# Number of frames per video
_images_per_file = 20

# Number of frames per training set
_num_images_train = _num_files_train * _images_per_file

# Video extension
video_exts = ".avi"

in_dir = "data"

#url of video stream
url = 'http://26.146.143.10:8080/video'
# url = 'http://26.72.110.56:8080/video'


if __name__ == "__main__":
    cap = cv2.VideoCapture(url)
    count = 0 
    images=[]
    shape = (_images_per_file,) + img_size_touple + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float16)
    while(True):
        ret, frame = cap.read()
        count+=1
        
        if count <= _images_per_file:
            RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = cv2.resize(RGB_img, dsize=(img_size, img_size),interpolation=cv2.INTER_CUBIC) 
            images.append(res)
        else:
            resul = np.array(images)
            resul = (resul / 255.).astype(np.float16)
            transfer_shape = (_images_per_file, transfer_values_size)
            transfer_values = np.zeros(shape=transfer_shape, dtype=np.float16)
            transfer_values = image_model_transfer.predict(image_batch)
            print(transfer_values.shape)
            inp = np.array(transfer_values)
            inp = inp[np.newaxis,...]
            print(inp.shape)
            pred = model.predict(inp)
            res = np.argmax(pred[0])
            count = 0
            images = []
            shape = (_images_per_file,) + img_size_touple + (3,)
            image_batch = np.zeros(shape=shape, dtype=np.float16)

            #print result
            if res == 0:
                print("\n\n"+ colored('VIOLENT','red')+" Video with confidence: "+str(round(pred[0][res]*100,2))+" %")
            else:
                print("\n\n" + colored('NON-VIOLENT','green') +" Video with confidence: "+str(round(pred[0][res]*100,2))+" %")
        # showing the video stream
        if frame is not None:
            cv2.imshow('frame',frame)
        q = cv2.waitKey(1)
        if q == ord("q"):
            break
    cv2.destroyAllWindows()



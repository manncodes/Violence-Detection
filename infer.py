import cv2
import os
import numpy as np
import keras
from keras.applications import VGG16
from keras import backend as K
from keras.models import Model
import sys
import time
from termcolor import colored

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



def get_frames(current_dir, file_name):
    in_file = os.path.join(current_dir, file_name)
    images = []
    vidcap = cv2.VideoCapture(in_file)
    success,image = vidcap.read()
    count = 0
    while count<_images_per_file:
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = cv2.resize(RGB_img, dsize=(img_size, img_size),interpolation=cv2.INTER_CUBIC) 
        images.append(res)
        success,image = vidcap.read()
        count += 1
    resul = np.array(images)
    resul = (resul / 255.).astype(np.float16)
    return resul


def get_transfer_values(current_dir, file_name):
    
    # Pre-allocate input-batch-array for images.
    shape = (_images_per_file,) + img_size_touple + (3,)
    image_batch = np.zeros(shape=shape, dtype=np.float16)
    image_batch = get_frames(current_dir, file_name)
    # Pre-allocate output-array for transfer-values.
    # Note that we use 16-bit floating-points to save memory.
    shape = (_images_per_file, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)

    transfer_values = \
            image_model_transfer.predict(image_batch)
            
    return transfer_values


def infer(curr_dir,file_name):
    tr = get_transfer_values(curr_dir,file_name)
    tr = tr[np.newaxis,...]
    pred = model.predict(np.array(tr))
    res = np.argmax(pred[0])
    if res == 0:
        print("\n\n"+ colored('VIOLENT','red')+" Video with confidence: "+str(round(pred[0][res]*100,2))+" %")
    else:
        print("\n\n" + colored('NON-VIOLENT','green') +" Video with confidence: "+str(round(pred[0][res]*100,2))+" %")


if __name__ == "__main__":
    arg = sys.argv
    model = keras.models.load_model('model/vlstm_92.h5')
    image_model = VGG16(include_top=True, weights='imagenet')    
    #We will use the output of the layer prior to the final
    # classification-layer which is named fc2. This is a fully-connected (or dense) layer.
    transfer_layer = image_model.get_layer('fc2')
    image_model_transfer = Model(inputs=image_model.input,outputs=transfer_layer.output)
    transfer_values_size = K.int_shape(transfer_layer.output)[1]
    start_time = time.time()
    infer(in_dir,arg[1])
    end_time = time.time()
    delta = round(end_time-start_time,2)
    print("Inferrence time: "+str(delta))
    print(str(round(20/delta,2))+" fps ^_^")

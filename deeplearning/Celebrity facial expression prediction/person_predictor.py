"""
Our person predictor file 
"""
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import numpy as np
import cv2

# define our image dimensions
img_width, img_height = 160, 160
# then load the weights
model_path = 'models/model_celebs.h5'
model_weights_path = 'models/weights_celebs.h5'
model = load_model(model_path)

def celeb_predict(file):
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    if answer == 0:
        label = "Angelina Jolie"
    elif answer == 1:
        label= "Barack Obama"
    elif answer == 2:
        label = "Donald Trump"
    elif answer == 3:
        label = "Gal Gadot"
    elif answer == 4:
        label ="Rihanna"
        
    return label

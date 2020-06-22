import numpy as np
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import load_img

face_cascade = cv2.CascadeClassifier('static/haarcascade_frontalface_default.xml')

from keras.models import model_from_json
model = model_from_json(open("models/facial_expression_model_structure.json", "r").read())
model.load_weights('models/facial_expression_model_weights.h5') #load weights

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


def detect(image_loader):
    image_load = cv2.imread(image_loader)
    gray = cv2.cvtColor(image_load, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    pred_list= []
    for (x,y,w,h) in faces:
        cv2.rectangle(image_load,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image
    
        detected_face = image_load[int(y):int(y+h), int(x):int(x+w)] #crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
    
        img_pixels = image.img_to_array(detected_face)

        img_pixels = np.expand_dims(img_pixels, axis = 0)
    
        img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
    
        predictions = model.predict(img_pixels) #store probabilities of 7 expressions
    
        max_index = np.argmax(predictions[0])
        
        emotion = emotions[max_index]
    
        percentage = round(predictions[0][max_index]*100, 2)
        
        pred_list = [emotion,percentage]
       
    return pred_list

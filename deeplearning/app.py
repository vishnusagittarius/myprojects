#import supporting packages
import os
 
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import time
import uuid
import base64
import sys

## gender classification 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib


from flask import send_from_directory
from werkzeug import SharedDataMiddleware

"""
We import our facial expression and person predictor files 
"""
from person_predictor import celeb_predict
from facial_expression import detect
import pandas as pd 

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)
def my_random_string(string_length=10):
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def gender_predictor(names):
    df = pd.read_csv('bonusclf/names_dataset.csv')
    df_X = df.name 
    df_Y = df.sex

    cv = CountVectorizer()
    X = cv.fit_transform(df_X)

    nb_model = open('bonusclf/nbmodel.pkl', 'rb')
    clf = joblib.load(nb_model)
    vect = cv.transform(names).toarray()
    my_prediction = clf.predict(vect)
    if my_prediction == 0:
        return 'Female'
    return 'Male'


app = Flask(__name__)
app.config['UPLOADER_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('layout.html', label='', imagesource='uploads/template.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOADER_FOLDER'], filename)
            file.save(file_path)

            expression = detect(file_path)
            
            emotion = expression[0]
            percentage = expression[1]
            
            result = celeb_predict(file_path)
            celebrity_name = result.split(' ')
            celebrity_name[0]
            celebrity_gender = gender_predictor([celebrity_name[0]])
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOADER_FOLDER'], filename))
            return render_template('layout.html',label=result,emotion=emotion, celeb_gender= celebrity_gender,percentage = percentage, imagesource='uploads/' + filename)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOADER_FOLDER'],
                               filename)


app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOADER_FOLDER']
})

if __name__ == "__main__":
    app.debug=False
    app.run()
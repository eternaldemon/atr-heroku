
#import base64
#import tkinter
import numpy as np
#import io
import cv2
import tensorflow as tf
from tensorflow import keras
#from keras.models import Sequential
#from keras.models import load_model
#from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import img_to_array
from flask import request,redirect,url_for,render_template
#from flask import jsonify
from flask import Flask
import matplotlib.image as mpimg 
#import matplotlib.pyplot as plt 
from werkzeug.utils import secure_filename
#from flask_opencv_streamer.streamer import Streamer
import pytesseract

from funct import coordinates,rescaleobj,rescaleocr
#from ocr1 import do_ocr,captureimage
import ctypes
user32 = ctypes.windll.user32
a,b = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
#print(a)
#print(b)             
app = Flask(__name__)


@app.route('/success')
def success():
    prediction=request.args.get('prediction')
    #touch_coordinates=request.args.get('touch_coordinates')
    coordinates2=rescaleobj(a,b)
    # touch_coordinates=coordinates
    name=request.args.get('name')
    global touch_coordinates
    global coordinates1
    touch_coordinates=coordinates(name)
    touch_coordinates[0][0]=int((float(touch_coordinates[0][0])/320.0)*a)
    touch_coordinates[0][1]=int((float(touch_coordinates[0][1])/240.0)*b)
    coordinates1=rescaleocr(a,b)
    #print(coordinates1[0][0])
    #print(coordinates1)
    x,y = touch_coordinates[0][0], touch_coordinates[0][1]
    #x,y=700,350
    x0,y0,x1,y1 = coordinates1[0][0],coordinates1[0][1],coordinates1[1][0],coordinates1[2][1]

    if float(prediction)<0:
        detected=0
    else:
        detected=1
    '''
    if float(prediction)<0:
        return render_template("index.html",detected=0,coordinates=coordinates2,coordinates1=coordinates1,touch_coordinates=touch_coordinates)
    else:
        return render_template("index.html",detected=1,coordinates1=coordinates1,coordinates=coordinates2,touch_coordinates=touch_coordinates)
    '''
    # if(detected and ((x <=x1 and x>=x0) and (y>=y0 and y<=y1)))
    # displayocr(detected,x,y,x0,y0,x1,y1)
    
    if(detected == 1):
        if((x <=x1 and x>=x0) and (y>=y0 and y<=y1)):
            return render_template("ocr.html",touch_coordinates=touch_coordinates,coordinates=coordinates1)
        else:
            return render_template("index.html",detected=1,coordinates1=coordinates1,coordinates=coordinates2,touch_coordinates=touch_coordinates)
    else:
        return render_template('index.html',detected=0,coordinates=coordinates2,coordinates1=coordinates1,touch_coordinates=touch_coordinates)

@app.route('/')
def fun1():
    return render_template("index.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename('temp.jpg'))
      image = mpimg.imread('temp.jpg')
      image=cv2.resize(image,(320,240))
      image=np.reshape(image,(1,240,320,3))
      image=image/255.0
      
      model=tf.keras.models.load_model('my_model.h5')
      prediction = model.predict(image)
      prediction=prediction[0][0]
      #print('Predicted')
      return redirect(url_for('success',prediction=prediction,name='temp.jpg'))


@app.route("/selectimage",methods=["GET","POST"])
def selected():
    x=request.form.get('selectimage')
    image = mpimg.imread(x)
    image = np.reshape(image,(1,240,320,3))
    image = image/255.0
    model = tf.keras.models.load_model('my_model.h5')
    prediction = model.predict(image)
    prediction = prediction[0][0]
    return redirect(url_for('success',prediction=prediction,name=x))


@app.route('/OCR',methods=["GET","POST"])
def Ocr():
    return render_template("ocr.html")

@app.route("/upload_ocr",methods=["GET","POST"])
def upload_ocr():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename('temp1.jpg'))
        img = cv2.imread('temp1.jpg',cv2.COLOR_BGR2GRAY)
        #pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        config = ('-l eng --oem 1 --psm 3')
        text = pytesseract.image_to_string(img, config=config) 
        text = text.split('\n')
        return render_template("ocr.html",text=text,touch_coordinates=touch_coordinates,coordinates=coordinates1)

@app.route('/go_back',methods=["GET","POST"])
def go_back():
    return render_template("index.html")

@app.route("/selectocrimage",methods=["GET","POST"])
def selected1():
    x=request.form.get('selectocrimage')
    img = cv2.imread(x,cv2.COLOR_BGR2GRAY)
    #pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    config = ('-l eng --oem 1 --psm 6')
    text = pytesseract.image_to_string(img, config=config) 
    text = text.split('\n')
    return render_template("ocr.html",text=text,touch_coordinates=touch_coordinates,coordinates=coordinates1)

if __name__=="__main__":
    app.run(debug=False)
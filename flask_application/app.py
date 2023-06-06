import os
import uuid
import flask
import urllib 
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request , send_file
from tensorflow.keras.preprocessing.image import load_img , img_to_array
from gevent.pywsgi import WSGIServer
import base64



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
age_model = load_model(os.path.join(BASE_DIR , 'cnn_models/age_best_model.h5'))
gender_model = load_model(os.path.join(BASE_DIR , 'cnn_models/gender_best_model.h5'))

ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])

classes_age = ["Moins de 20 ans", "Entre 20 et 40 ans", "Entre 40 et 60 ans", "Entre 60 et 80 ans", "Plus de 80 ans"]
classes_gender = ["Homme", "Femme"]

def predict_age(filename, model):
    prediction_img = load_img(filename , target_size = (224 , 224))
    prediction_img = img_to_array(prediction_img)
    prediction_img = prediction_img.reshape(1, 224, 224, 3)

    prediction_img = prediction_img.astype('float32')
    prediction_img = prediction_img/255.0
    result = age_model.predict(prediction_img)[0]

    dict_result = {}
    for i in range(5):
        dict_result[result[i]] = classes_age[i]
    
    result.sort()
    result = result[::-1]
    result = result[:3]
    
    prob_result = []
    class_result = []
    
    for i in range(3):
        prob_result.append((result[i]*100).round(2))
        class_result.append(dict_result[result[i]])

    return class_result , prob_result

def predict_gender(filename, model):
    prediction_img = load_img(filename , target_size = (224 , 224))
    prediction_img = img_to_array(prediction_img)
    prediction_img = prediction_img.reshape(1, 224, 224, 3)

    prediction_img = prediction_img.astype('float32')
    prediction_img = prediction_img/255.0
    #result = gender_model.predict(prediction_img)[0]
    result = gender_model.predict(prediction_img)

    """dict_result = {}
    for i in range(2):
        dict_result[result[i]] = classes_gender[i]
    
    result.sort()
    result = result[::-1]
    
    prob_result = []
    class_result = []
    
    for i in range(2):
        prob_result.append((result[i]*100).round(2))
        class_result.append(dict_result[result[i]])

    return class_result , prob_result"""
    return result
    


app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT



@app.route('/')
def home():
        return render_template('index.html')
    
    
@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ""
    target_img = os.path.join(os.getcwd() , 'static/images')
    
    if request.method == 'POST':
        if(request.form):
            img_data = request.form['file']
            file_number = len(os.listdir(target_img))
            img = "image"+str(file_number)+".png"
            file_path = target_img + "/" + img
            
            if os.path.isfile(file_path):
                os.remove(file_path)
                
            with open(file_path, "wb") as fh:
                fh.write(base64.decodebytes(img_data.encode()))
            
            class_result_age, prob_result_age = predict_age(file_path, age_model)

            predictions_age = {
                'class1':class_result_age[0],
                'class2':class_result_age[1],
                'class3':class_result_age[2],
                'prob1': prob_result_age[0],
                'prob2': prob_result_age[1],
                'prob3': prob_result_age[2],
            }
            
            if predict_gender(file_path, gender_model) < 0.5:
                return  render_template('success.html', img = img, predictions_age = predictions_age, prediction_gender = "Homme")
            else:
                return  render_template('success.html', img = img, predictions_age = predictions_age, prediction_gender = "Femme")
            
        elif (request.files):
            file = request.files['file']
            
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename

                class_result_age, prob_result_age = predict_age(img_path, age_model)

                predictions_age = {
                    'class1':class_result_age[0],
                    'class2':class_result_age[1],
                    'class3':class_result_age[2],
                    'prob1': prob_result_age[0],
                    'prob2': prob_result_age[1],
                    'prob3': prob_result_age[2],
                }
                
            else:
                error = 'Please upload images of jpg , jpeg and png extension only'

            if(len(error) == 0):
                if predict_gender(img_path, gender_model) < 0.5:
                    return  render_template('success.html', img = img, predictions_age = predictions_age, prediction_gender = "Homme")
                else:
                    return  render_template('success.html', img = img, predictions_age = predictions_age, prediction_gender = "Femme")
                
            else:
                return render_template('index.html')

    else:
        return render_template('index.html')



if __name__ == "__main__":
    app.run()
    """http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()"""

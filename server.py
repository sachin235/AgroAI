import os
import random
import string
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from cmd_wheat_quality_detector_v2 import predict

UPLOAD_FOLDER = os.path.join(os.getcwd(), "static")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def random_string(stringLength=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_extension(filename):
    return filename.rsplit('.', 1)[1].lower()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            
            # save uploaded file to static with a random name
            filename = secure_filename(file.filename)
            ex = get_extension(filename)
            new_filename = random_string(10) + "." + ex
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
            
            print("Predicting")
            good, not_good = predict(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))
            print("Work done", good, not_good)
    
            # return payload
            return "Upload successful: " + url_for('uploaded_file', filename=new_filename)

    elif request.method == 'GET':
        return "AgroAI API"

# Route to fetch uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
from flask import Flask, request, redirect, url_for, render_template
from tensorflow_core.python.keras.saving.save import load_model
from werkzeug import secure_filename
from PIL import ImageOps, Image
import os
import numpy as np

app = Flask(__name__)

model = load_model('model.h5')


@app.route('/', methods=['GET', 'POST'])
def index_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(filename)
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')


@app.route('/prediction/<filename>')
def prediction(filename):
    img = Image.open(filename)
    img = ImageOps.grayscale(img)
    img = img.resize((28, 28))
    img = np.array(img)
    img = np.reshape(img, newshape=(1, 28, 28, 1))
    category = model.predict_classes(img)
    charter = 'a'
    if category >= 9:
        category = category-1
    charter = chr(category + 64)
    return render_template('predict.html', category=category,
                           char=charter)


if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
model = tf.keras.models.load_model("mlp_mnist_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the POST request
    img_data = request.form['image']
    img_data = img_data.split(',')[1]  # remove 'data:image/png;base64,' part
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('L')  # grayscale
    img = img.resize((28,28))
    img = np.array(img)
    img = 255 - img  # invert colors (white background to black)
    img = img.reshape(1,784) / 255.0

    prediction = model.predict(img)
    digit = int(np.argmax(prediction))
    return jsonify({"Predicted_Digit": digit})  # <-- key has no spaces

if __name__ == '__main__':
    app.run(debug=True)

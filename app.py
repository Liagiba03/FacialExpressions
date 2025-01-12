from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Cargar modelos
facial_points_model = load_model('models/model_keyfacial.h5')
emotion_model = load_model('models/model_facialexpression.h5')

# Diccionario para mapear etiquetas a emociones
label_to_text = {0: 'Ira', 1: 'Odio', 2: 'Miedo', 3: 'Felicidad', 4: 'Tristeza', 5: 'Sorpresa', 6: 'Neutral'}

def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size, color_mode='grayscale')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def guardar_imagen_procesada(file, facial_points):
    img = Image.open(file)
    img = img.resize((96, 96))  # Redimensionar la imagen a 96x96
    img = img.convert('RGB')  # Convertir a escala de grises

    # Asegurarse de que los puntos estén en el formato correcto (como lista plana)
    if isinstance(facial_points, str):
        facial_points = [float(x) for x in facial_points.strip('[]').split()]

    # Dibujar los puntos faciales en la imagen
    draw = ImageDraw.Draw(img)
    for i in range(0, len(facial_points), 2):
        x = facial_points[i] * img.width / 20  # Ajustar coordenadas x
        y = facial_points[i + 1] * img.height / 20  # Ajustar coordenadas y
        draw.ellipse((x-2, y-2, x+2, y+2), fill='red', outline='red')


    img_base64 = image_to_base64(img) # Convertir a base64
    # Guardar la imagen redimensionada
    #img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + file.filename)
    #img.save(img_path)
    return img_base64

def image_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('ascii')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Preprocesar imagen para puntos faciales
            img_keyfacial = preprocess_image(file_path, target_size=(96, 96))

            # Predecir puntos faciales
            facial_points = facial_points_model.predict(img_keyfacial)[0]

            # IMAGEN PROCESADA
            img_base64 = guardar_imagen_procesada(file, facial_points)

            # Preprocesar imagen para emociones faciales
            img_facialexpression = preprocess_image(file_path, target_size=(48, 48))

            # Predecir emoción
            emotion = emotion_model.predict(img_facialexpression)
            emotion_label = np.argmax(emotion)
            emotion_text = label_to_text[emotion_label]

            return render_template('result.html', facial_points=facial_points, emotion_label=emotion_text, image_base64=img_base64)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

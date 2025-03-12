import tensorflow as tf

model = tf.keras.models.load_model('D:\\project_rasp\\emnist_letter_recognition_model.h5', compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_quantized_model = converter.convert()
with open('D:\\project_rasp\\emnist_letter_recognition_model.tflite', 'wb') as f:f.write(tflite_quantized_model)
import numpy as np
import tensorflow as tf
from PIL import Image

def load_tflite_model(file_path):
    interpreter = tf.lite.Interpreter(model_path=file_path)
    interpreter.allocate_tensors()
    return interpreter
#Przetwarzanie obrazów i predykcja
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L') # Konwersja na skale szarości
    img = img.resize((28, 28), Image.ANTIALIAS) # Zmiana rozmiaru na 28x28
    img = np.array(img)
    img = img.reshape(1,

    28, 28, 1) # Zmiana kształtu do formatu sieci
    img = img / 255.0 # Normalizacja
    return img

def predict_image(image_path, model):
    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image)
    return np.argmax(prediction)

def predict_image_tflite(image_path, interpreter):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        processed_image = preprocess_image(image_path)
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        return np.argmax(output_data) 

# Ścieżka do kwantyzowanego modelu TensorFlow Lite
tflite_model_path = 'D:\\project_rasp\\emnist_letter_recognition_model.tflite'
interpreter = load_tflite_model(tflite_model_path)

# Przewidywanie klasy
image_path = 'D:\\project_rasp\\image\\image_1.jpg'
predicted_class = predict_image_tflite(image_path, interpreter)
print(f"Predicted class: {predicted_class}")

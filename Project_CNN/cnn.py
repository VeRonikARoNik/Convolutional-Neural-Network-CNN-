import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from emnist import extract_training_samples, extract_test_samples
from PIL import Image
from tensorflow.keras.models import load_model

#Wczytywanie i przetwarzanie danych EMNIST
x_train, y_train = extract_training_samples('letters')
x_test, y_test = extract_test_samples('letters')
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = to_categorical(y_train - 1, num_classes=26)
y_test = to_categorical(y_test - 1, num_classes=26)

#Budowanie uproszczonego modelu CNN
model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(26, activation='softmax')) # 26 klas dla liter

#Kompilacja modelu
model.compile(optimizer='adam',
loss='categorical_crossentropy', metrics=['accuracy'])

#Trenowanie modelu z mniejszym rozmiarem partii
model.fit(x_train, y_train, epochs=1, batch_size=5, validation_split=0.2)

#Ewaluacja
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

#Zapis
model.save('D:\\project_rasp\\emnist_letter_recognition_model.h5')

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

#Wczytaj model raz
loaded_model = load_model('D:\\project_rasp\\emnist_letter_recognition_model.h5')

#Przewidywanie klasy
image_path = 'D:\\project_rasp\\image\\image_1.jpg'
predicted_class = predict_image(image_path, loaded_model)
print(f"Predicted class: {predicted_class}")
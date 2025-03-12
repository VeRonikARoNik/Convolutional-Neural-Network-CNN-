import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from emnist import extract_training_samples, extract_test_samples
from PIL import Image
from tensorflow.keras.models import load_model

# Wczytywanie i przetwarzanie danych EMNIST
x_train, y_train = extract_training_samples('letters')
x_test, y_test = extract_test_samples('letters')
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = to_categorical(y_train - 1, num_classes=26)
y_test = to_categorical(y_test - 1, num_classes=26)

# Budowanie modelu CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(26, activation='softmax'))  # 26 klas dla liter

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy'])

#trenowanie modelu
model.fit(x_train, y_train, epochs=1, batch_size=5, validation_split=0.2)

#ewaluacja
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

#zapis
model.save('D:\\project_rasp\\emnist_letter_recognition_model.h5')

#ładowanie modelu
loaded_model = load_model('D:\\project_rasp\\emnist_letter_recognition_model.h5')
#przetważanie
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L') # Konwersja na skale szarości
    img = img.resize((28, 28), Image.ANTIALIAS) # Zmiana rozmiaru na 28x28
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1) # Zmiana kształtu do formatu sieci
    img = img / 255.0 # Normalizacja
    return img

def predict_image(image_path):
    class_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                       10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
                       20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
    processed_image = preprocess_image(image_path)
    #loaded_model = load_model('D:\\project_rasp\\emnist_letter_recognition_model.h5')
    prediction = loaded_model.predict(processed_image)
    predicted_class_number = np.argmax(prediction)
    predicted_letter = class_to_letter[predicted_class_number]
    return predicted_letter

image_path ='D:\\project_rasp\\image\\image_1.jpg'
predicted_class = predict_image(image_path)
print(f"Predicted class: {predicted_class}")
predicted_letter = predict_image(image_path, loaded_model)
print(f"Predicted letter: {predicted_letter}")
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import os
import pandas as pd

DIR_IMAGES = r'--Your Dir'

img1 = 'img1.jpeg'
img2 = 'img2.png'
img3 = 'img2.png'
img4 = 'img2.jpeg'

# Загружаем обученную модель
model = load_model(r'--Your Model.h5')

def predict(img_path):
    # Загружаем изображение и приводим его к нужному размеру
    img = load_img(img_path, target_size=(200, 200))

    # Преобразуем изображение в массив numpy
    img_array = img_to_array(img)

    # Нормализуем пиксели изображения
    img_array = preprocess_input(img_array)

    # Добавляем размерность для пакета изображений (batch)
    img_array = np.expand_dims(img_array, axis=0)

    # Получаем предсказание модели
    prediction = model.predict(img_array)

    # Выводим результаты предсказания
    if (prediction[0][0] == 0):
        return('handed')
    else:
        return ('machine')

# predict(os.path.join(DIR_IMAGES,img1))
# print ()
# predict(os.path.join(DIR_IMAGES,img2))
# print()
# predict(os.path.join(DIR_IMAGES,img3))
# print ()
# predict(os.path.join(DIR_IMAGES,img4))

dat = []

for filename in os.listdir(r'--Some tests'):
    filename = os.path.join(DIR_IMAGES,filename)
    
    dat.append([filename, predict(filename)])

df = pd.DataFrame(dat)
df.to_csv(r'Results.csv')


import tensorflow as tf
from torchvision import datasets
import torchvision.transforms as transforms
import torch
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_HEIGHT = 200
IMG_WIDTH = 200

train_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    r'train',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=4,
    class_mode='binary')



model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Компилируем модель
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучаем модель
model.fit(train_data, epochs=1)


model.save(r'model.h5')


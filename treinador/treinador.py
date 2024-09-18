import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import pathlib
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

url = './modelo-tombamento/dataset_37x37/'

data_dir = pathlib.Path(url)

# print(len(list(data_dir.glob('*/*.jpg'))))

subfolders = [f.name for f in data_dir.iterdir() if f.is_dir()]

# print(subfolders)

latatombada = list(data_dir.glob('empe/*'))

def plota_resultados(history,epocas):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    intervalo_epocas = range(epocas)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(intervalo_epocas, acc, label='Acurácia do Treino')
    plt.plot(intervalo_epocas, val_acc, label='Acurácia da Validação')
    plt.legend(loc='lower right')


    plt.subplot(1, 2, 2)
    plt.plot(intervalo_epocas, loss, label='Custo do Treino')
    plt.plot(intervalo_epocas, val_loss, label='Custo da Validação')
    plt.legend(loc='upper right')
    plt.show()

for subfolder in subfolders:
  path = data_dir / subfolder
  images = list(path.glob('*.jpg'))
  print(f'classe {subfolders} tem {len(images)} imagens')

  if images:
    img = Image.open(str(images[0]))
    img_array = np.array(img)
    print(f'Dimensão da primeira image em {subfolder}: {img_array.shape}')

batch_size = 64

altura = 150
largura = 150

shape = (altura, largura, 3)

treino = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=568,
    image_size=(altura,largura),
    batch_size=batch_size
)

validacao = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=568,
    image_size=(altura,largura),
    batch_size=batch_size
)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.93):
      print("\n Alcançamos 93% de acurácia. Parando o treinamento!")
      self.model.stop_training = True

callbacks = myCallback()

data_augmentation = tf.keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.05),
  ]
)

modelo = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=shape),
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    # Add convolutions and max pooling
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(),
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
    )

epocas = 50

history = modelo.fit(
    treino,
    validation_data=validacao,
    epochs=epocas
)

modelo.summary()

modelo.save('./modelo-tombamento')

plota_resultados(history,epocas)


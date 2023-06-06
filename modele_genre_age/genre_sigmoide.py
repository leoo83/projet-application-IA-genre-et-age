


from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.models import load_model
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.utils import shuffle

import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

BATCH_SIZE = 32
IMAGE_SIZE = [224,224]
BASE_DIR = 'C:\\Users\\hanif\\OneDrive\\Bureau\\CNN\\copie'

# read labels of image age, gender
image_paths = []
age_labels = []
gender_labels = []

for filename in os.listdir(BASE_DIR):
  # check if the file is an image
  if not filename.endswith('.jpg'):
    continue

  # read image path
  image_path = os.path.join(BASE_DIR, filename)

  # split filename to extract labels
  temp = filename.split('_')

  # check if filename has the correct format
  if len(temp) != 4:
    print(f"Warning: incorrect filename format for {filename}")
    continue

  try:
    age = int(temp[0])
    gender = int(temp[1])
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)
  except ValueError:
    print(f"Warning: incorrect age or gender value for {filename}")

# extract name of file
names = []
for path in image_paths:
    temp = path.split('\\')
    names.append(temp[7])

# convert to dataframe
df = pd.DataFrame()
df['image'], df['age'], df['gender'], df[
  'name'] = image_paths, age_labels, gender_labels,  names

print(df['gender'])


# split dataset into train, validation, and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

# set up data generators
train_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess_input)




train_generator = train_datagen.flow_from_dataframe(train_df, directory=BASE_DIR,
                                                    x_col='name', y_col='gender',
                                                    target_size=IMAGE_SIZE,
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='raw')
val_generator = val_datagen.flow_from_dataframe(val_df, directory=BASE_DIR,
                                                x_col='name', y_col='gender',
                                                target_size=IMAGE_SIZE,
                                                batch_size=BATCH_SIZE,
                                                class_mode='raw')

test_generator = test_datagen.flow_from_dataframe(test_df, directory=BASE_DIR,
                                                  x_col='name', y_col='gender',
                                                  target_size=IMAGE_SIZE,
                                                  batch_size=BATCH_SIZE,
                                                  class_mode='raw')




from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# charger le modèle VGG16 pré-entraîné sans la dernière couche dense
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# geler les poids des couches de base (sauf les dernières)
for layer in base_model.layers[:-4]:
    layer.trainable = False

# ajouter des couches supplémentaires pour la classification
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# construire le modèle final en joignant le modèle de base et les couches supplémentaires
model = Model(inputs=base_model.input, outputs=predictions)

# compiler le modèle avec une fonction de perte binaire_crossentropy et un optimiseur Adam
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# définir le nombre d'itérations par époque et les tailles de lot pour l'entraînement et la validation
batch_size = 32
steps_per_epoch = train_generator.n // batch_size
validation_steps = val_generator.n // batch_size
NR_EPOCHS = 3

# entraîner le modèle
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=NR_EPOCHS,
    validation_data=val_generator,
    validation_steps=validation_steps,
    verbose=1
)


# spécifier le chemin de sauvegarde du modèle
model_path ='C:\\Users\\hanif\\OneDrive\\Bureau\\CNN'

# sauvegarder le modèle
model.save(model_path)

print(history.history.keys())
import matplotlib.pyplot as plt


# Charger les données d'entraînement et de validation
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']  # Utilisez 'accuracy' au lieu de 'acc'
val_acc = history.history['val_accuracy']  # Utilisez 'val_accuracy' au lieu de 'val_acc'

# Tracer la courbe de la perte d'entraînement et de validation
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Tracer la courbe de l'exactitude d'entraînement et de validation
plt.plot(train_acc, label='Train Accuracy')
plt.plot(val_acc, label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Charger le modèle
modell = tf.keras.models.load_model(model_path)
# évaluer le modèle sur l'ensemble de test
test_loss, test_acc = modell.evaluate(test_generator, verbose=0)
print('Test accuracy:', test_acc)


# Make predictions on the test set
predictions = modell.predict(test_generator)
print(predictions)
# Convert predictions to binary values (0 or 1)
binary_predictions = np.round(predictions)
print(binary_predictions)
# Get the true labels from the testing_data DataFrame
y_true = list(test_df['gender'])


# Create a confusion matrix
confusion_matrix = tf.math.confusion_matrix(y_true, binary_predictions)

# Convert the confusion matrix to a NumPy array
confusion_matrix = confusion_matrix.numpy()

# Print the confusion matrix
print(confusion_matrix)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt

import itertools
# Afficher les résultats

# Calculer le F1-score, le rappel, la précision et le support
f1 = f1_score(y_true, binary_predictions)
recall = recall_score(y_true, binary_predictions)
precision = precision_score(y_true, binary_predictions)
support = np.bincount(y_true)
print("F1-score:", f1)
print("Rappel:", recall)
print("Précision:", precision)
print("Support:", support)


# Impression des prédictions et des véritables étiquettes
for i in range(len(binary_predictions)):
    prediction = binary_predictions[i]
    true_label = y_true[i]
    print("Prédiction:", prediction)
    print("Véritable étiquette:", true_label)
    print("---")



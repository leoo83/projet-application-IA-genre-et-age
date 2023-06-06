
from sklearn.datasets import images
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

from tqdm.notebook import tqdm



from PIL import Image
import shutil

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

IMAGE_SIZE = [224,224]
TRAIN_TEST_SPLIT = 0.3
BASE_DIR = 'C:\\Users\\axlco\\OneDrive\\Bureau\\CNN\\UTKFace'
NR_EPOCHS = 15

def age_to_class( age ):
  age_class = None
  # 0 (0-20)
  if 0 <= age <= 20:
    age_class = 0
  # 1 (20-40)
  if 20 < age <= 40:
    age_class = 1
  # 2 (40-60)
  if 40 < age <= 60:
    age_class = 2
  # 3 (40-60)
  if 60 < age <= 80:
    age_class = 3
  # 3 (80+)
  if age > 80:
    age_class = 4
  return age_class

# map age classes for age gaps
age_dict = {0:'0-20', 1:'20-40',2:'40-60', 3:'60-80', 4:'80+'}

def decode_age_label(age_id):
  return age_dict[int(age_id)]


# read labels of image age, gender
image_paths = []
age_labels = []
gender_labels = []
age_class = []

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
    age_class.append(age_to_class(age))
  except ValueError:
    print(f"Warning: incorrect age or gender value for {filename}")

# extract name of file
names = []
for path in image_paths:
    temp = path.split('\\')
    names.append(temp[7])

# convert to dataframe
df = pd.DataFrame()
df['image'], df['age'], df['gender'], df['age_class'], df[
  'name'] = image_paths, age_labels, gender_labels, age_class, names

# get and print the number of images in dataframe
age_verbose = [age_dict[int(age)] for age in df['age_class']]
print(Counter(age_verbose))


# shuffle dataset
df = shuffle(df)

# check dataframe
print(df)

training_data = df.sample(frac=0.8, random_state=25)
temp = df.drop(training_data.index)

val_data = temp.sample(frac=0.7, random_state=25)
testing_data = temp.drop(val_data.index)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of training examples: {val_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")


 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1./255,
                               preprocessing_function=preprocess_input)

val_datagen = ImageDataGenerator(rescale=1./255,
                                preprocessing_function=preprocess_input)


CLASS_MODE = 'raw'
BATCH_SIZE = 64
BASE_DIR = 'C:\\Users\\axlco\\OneDrive\\Bureau\\CNN\\UTKFace'

train_generator = train_datagen.flow_from_dataframe(dataframe=training_data,
                                                directory=BASE_DIR,
                                                x_col='name',
                                                y_col='age_class',
                                                class_mode=CLASS_MODE,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                target_size=IMAGE_SIZE)



val_generator = val_datagen.flow_from_dataframe(dataframe=val_data,
                                              directory=BASE_DIR,
                                              x_col='name',
                                              y_col='age_class',
                                              class_mode=CLASS_MODE,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              target_size=IMAGE_SIZE)

test_generator = val_datagen.flow_from_dataframe(dataframe=testing_data,
                                              directory=BASE_DIR,
                                              x_col='name',
                                              y_col='age_class',
                                              class_mode=CLASS_MODE,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              target_size=IMAGE_SIZE)


import tensorflow as tf
from keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

from pathlib import Path
path = Path("./models")
path.mkdir(exist_ok=True)


MODEL_PATH = "C:\\Users\\axlco\\OneDrive\\Bureau\\CNN\\age\\age_best_model.h5"


checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_PATH,
                                                monitor='val_sparse_categorical_accuracy',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='max')

import datetime
log_dir = 'C:\\Users\\axlco\\OneDrive\\Bureau\\CNN\\age_model_' + datetime.datetime.now().strftime('%y-%m-%d_%H-%M')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)
from keras.applications.inception_v3 import InceptionV3, preprocess_input


# load VGG16 without the last 3 Dense layers, in order to train these new
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# check if input type is correct
vgg.inputs

# all layers before are locked
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(4096, activation='relu')(x)
x = Dense(4096, activation='relu')(x)
prediction = Dense(5, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.summary()


# with tpu_strategy.scope():
#     x = Flatten()(vgg.output)
#     x = Dense(4096, activation='relu')(x)
#     x = Dense(4096, activation='relu')(x)
#     prediction = Dense(5, activation='softmax')(x)
#     model = Model(inputs=vgg.input, outputs=prediction)

#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#               metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
tf.keras.utils.plot_model
from keras import callbacks

earlystopping = callbacks.EarlyStopping(monitor ="val_loss",
                                        mode ="min", patience = 5,
                                        restore_best_weights = True)

history = model.fit(train_generator,
                    epochs=15,
                    validation_data=val_generator,
                    callbacks = [tensorboard_callback, checkpoint]
                    )

# print keys
history.history.keys()


NR_EPOCHS = len(history.history['loss'])
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(NR_EPOCHS)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

accuracy_train = history.history['sparse_categorical_accuracy']
accuracy_val = history.history['val_sparse_categorical_accuracy']
epochs = range(NR_EPOCHS)
plt.plot(epochs, accuracy_train, 'g', label='Training accuracy')
plt.plot(epochs, accuracy_val, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model = load_model("C:\\Users\\axlco\\OneDrive\\Bureau\\CNN\\age_resnet\\age_best_model.h5")
loss, acc = model.evaluate(test_generator)
print (loss, acc)



# Predict the age classes of the test set
# convert labels to list
y_true = list(testing_data['age_class'])
y_pred = model.predict(test_generator)

# Convert the predicted probabilities to class labels
y_pred = np.argmax(y_pred, axis=1)

# Create the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=age_dict.values(), yticklabels=age_dict.values())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Age Classification Model')
plt.show()











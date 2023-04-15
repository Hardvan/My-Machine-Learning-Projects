# %% [markdown]
# # Emotion Detection

# %% [markdown]
# ## Importing Libraries

# %%
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, Activation, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adamax
from keras.callbacks import ModelCheckpoint, EarlyStopping
import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import plotly.express as px
from keras.regularizers import l1, l2
from sklearn.metrics import accuracy_score
from keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from keras import regularizers
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
import warnings
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import pyplot
import os
import keras
import seaborn as sns
import random
import scikitplot
import numpy as np
import pandas as pd
!pip install scikit-plot

# %%
# Import ImageDataGenerator to generate images from the dataset

warnings.simplefilter("ignore")

# %% [markdown]
# ## Loading the Dataset

# %%
data = pd.read_csv('fer2013.csv')
data.head()

# %%
data.shape

# %%
data.isnull().sum()

# %% [markdown]
# ## Data Preprocessing

# %%
CLASS_LABELS = ['Anger', 'Disgust', 'Fear',
                'Happy', 'Neutral', 'Sadness', "Surprise"]
fig = px.bar(x=CLASS_LABELS,
             y=[list(data['emotion']).count(i)
                for i in np.unique(data['emotion'])],
             color=np.unique(data['emotion']),
             color_continuous_scale="Emrld")
fig.update_xaxes(title="Emotions")
fig.update_yaxes(title="Number of Images")
fig.update_layout(showlegend=True,
                  title={
                      'text': 'Train Data Distribution ',
                      'y': 0.95,
                      'x': 0.5,
                      'xanchor': 'center',
                      'yanchor': 'top'})
fig.show()

# %% [markdown]
# ### Shuffling the Data

# %%
data = data.sample(frac=1)

# %% [markdown]
# ### One Hot Encoding

# %%
labels = to_categorical(data[['emotion']], num_classes=7)

# %%
train_pixels = data["pixels"].astype(str).str.split(" ").tolist()
train_pixels = np.uint8(train_pixels)

# %% [markdown]
# ### Standardizing the Data

# %%
pixels = train_pixels.reshape((35887*2304, 1))


# %%
scaler = StandardScaler()
pixels = scaler.fit_transform(pixels)

# %% [markdown]
# ### Reshape the Data

# %%
pixels = train_pixels.reshape((35887, 48, 48, 1))


# %%
X_train, X_test, y_train, y_test = train_test_split(
    pixels, labels, test_size=0.1, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, shuffle=False)

# %%
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

# %%
plt.figure(figsize=(15, 23))
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
              3: 'Happiness', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
i = 1
for i in range(7):
    img = np.squeeze(X_train[i])
    plt.subplot(1, 7, i+1)
    plt.imshow(img)
    index = np.argmax(y_train[i])
    plt.title(label_dict[index])
    plt.axis('off')
    i += 1
plt.show()

# %% [markdown]
# ### Data augmentation using ImageDataGenerator

# %%
datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             horizontal_flip=True,
                             zoom_range=0.2)
valgen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            horizontal_flip=True,
                            zoom_range=0.2)

# %%
datagen.fit(X_train)
valgen.fit(X_val)

# %%
train_generator = datagen.flow(X_train, y_train, batch_size=64)
val_generator = datagen.flow(X_val, y_val, batch_size=64)

# %% [markdown]
# ## Building the Model

# %%


def cnn_model():

    model = tf.keras.models.Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
              activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu',
              kernel_regularizer=regularizers.l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(7, activation='softmax'))
    model.compile(
        optimizer=Adam(lr=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


# %%
model = cnn_model()


# %%
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# %%
model.summary()

# %%
checkpointer = [EarlyStopping(monitor='val_accuracy', verbose=1,
                              restore_best_weights=True, mode="max", patience=5),
                ModelCheckpoint('best_model.h5', monitor="val_accuracy", verbose=1,
                                save_best_only=True, mode="max")]

# %%
history = model.fit(train_generator,
                    epochs=10,
                    batch_size=64,
                    verbose=1,
                    callbacks=[checkpointer],
                    validation_data=val_generator)

# %% [markdown]
# ## Visualizing the Results

# %%
plt.plot(history.history["loss"], 'r', label="Training Loss")
plt.plot(history.history["val_loss"], 'b', label="Validation Loss")
plt.legend()

# %%
plt.plot(history.history["accuracy"], 'r', label="Training Accuracy")
plt.plot(history.history["val_accuracy"], 'b', label="Validation Accuracy")
plt.legend()

# %%
loss = model.evaluate(X_test, y_test)
print("Test Acc: " + str(loss[1]))

# %%
preds = model.predict(X_test)
y_pred = np.argmax(preds, axis=1)

# %%
label_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
              3: 'Happiness', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(X_test.shape[0], size=24, replace=False)):
    ax = figure.add_subplot(4, 6, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(X_test[index]))
    predict_index = label_dict[(y_pred[index])]
    true_index = label_dict[np.argmax(y_test, axis=1)[index]]

    ax.set_title("{} ({})".format((predict_index),
                                  (true_index)),
                 color=("green" if predict_index == true_index else "red"))

# %%
CLASS_LABELS = ['Anger', 'Disgust', 'Fear',
                'Happy', 'Neutral', 'Sadness', "Surprise"]

cm_data = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
cm = pd.DataFrame(cm_data, columns=CLASS_LABELS, index=CLASS_LABELS)
cm.index.name = 'Actual'
cm.columns.name = 'Predicted'
plt.figure(figsize=(15, 10))
plt.title('Confusion Matrix', fontsize=20)
sns.set(font_scale=1.2)
ax = sns.heatmap(cm, cbar=False, cmap="Blues", annot=True,
                 annot_kws={"size": 16}, fmt='g')

# %%
print(classification_report(np.argmax(y_test, axis=1), y_pred, digits=3))

# %% [markdown]
# ## Saving the Model

# %%
# Save the model
model.save(f'emotion_detector_{time.time()}.h5')

# %%

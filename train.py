import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from sklearn.model_selection import train_test_split

import numpy as np
import sklearn.metrics
import librosa
import os
import itertools
from matplotlib import pyplot as plt

np.random.seed(42)

gtzan = '/home/fran/tesis/datasets/gtzan/'
genre_list = ['metal', 'reggae', 'classical', 'country', 'hiphop']

SAVE_NPY = True
LOAD_NPY = True


def getdata():
    # Structure for the array of songs
    song_data = []
    genre_data = []

    # Read files from the folders
    for x in genre_list:
        for root, subdirs, files in os.walk(gtzan + x):
            for file in files:
                # Read the audio file
                file_name = gtzan + x + "/" + file
                print(file_name)
                signal, sr = librosa.load(file_name, duration=9)

                # Calculate the melspectrogram of the audio and use log scale
                melspec = librosa.feature.melspectrogram(
                    signal, sr=sr, n_fft=1024, hop_length=512)[:384]
                log_S = librosa.logamplitude(melspec, ref_power=np.max)

                # Append the result to the data structure
                song_data.append(log_S)
                genre_data.append(genre_list.index(x))

    return np.array(song_data), keras.utils.to_categorical(
        genre_data, len(genre_list))


# Keras Model
def genre_classification_baby(input_shape=(128, 130), nb_genres=10):
    model = Sequential()

    # 1
    model.add(
        Conv1D(
            filters=128,
            kernel_size=3,
            input_shape=input_shape,
            activation='relu',
            kernel_initializer='normal',
            padding='valid'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=2))
    model.add(Dropout(0.25))

    # 2
    model.add(
        Conv1D(
            filters=128,
            kernel_size=3,
            activation='relu',
            kernel_initializer='normal',
            padding='valid'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    # Regular MLP
    model.add(
        Dense(
            128,
            kernel_initializer='glorot_normal',
            bias_initializer='glorot_normal',
            activation='relu'))
    model.add(Dense(nb_genres, activation='softmax'))

    return model


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if SAVE_NPY:
    songs, genres = getdata()
    np.save(gtzan + 'songs.npy', songs)
    np.save(gtzan + 'genres.npy', genres)

if LOAD_NPY:
    songs = np.load(gtzan + 'songs.npy')
    genres = np.load(gtzan + 'genres.npy')

np.random.seed(42)

# Split DATA
x_train, x_test, y_train, y_test = train_test_split(
    songs, genres, test_size=0.1, stratify=genres)

# Split training set into training and validation
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=1 / 6, stratify=y_train)

# Model
model = genre_classification_baby(
    input_shape=(128, 388), nb_genres=5)

# Optimizers
sgd = keras.optimizers.SGD(lr=0.001)
adam = keras.optimizers.Adam(
    lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)

# Earlystop
earlystop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, verbose=0)

# Compiler for the model
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=sgd,
    metrics=['accuracy'])

# Fit
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    validation_data=(x_val, y_val),
    epochs=45,
    verbose=2,
    callbacks=[earlystop])

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_acc_classification.png', dpi=300)
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_loss_classification.png', dpi=300)
plt.show()

# Evaluate
score = model.evaluate(x_test, y_test, verbose=0)
score_val = model.evaluate(x_val, y_val, verbose=0)

# Print metrics
print('Test accuracy:', score[1])
print('Val accuracy: ', score_val[1])

# Plot confusion matrix
y_pred = model.predict_classes(x_test)
y_true = []

for y in y_test:
    genre = np.argmax(y)
    y_true.append(genre)

cnf_matrix = sklearn.metrics.confusion_matrix(y_pred, y_true)

# Plot normalized confusion matrix
plot_confusion_matrix(
    cnf_matrix,
    classes=genre_list,
    normalize=True,
    title='Normalized confusion matrix')
plt.tight_layout()
plt.savefig('confusion_matrix_classification.png', dpi=300)
plt.show()

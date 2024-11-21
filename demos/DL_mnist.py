# conda create -n mnist python=3.12.7
# conda activate mnist
# pip install keras==3.6.0 tensorflow==2.16.2
# pip install matplotlib==3.8.4
# python DL_mnist.py

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import to_categorical

'''
    Digit recognition on the MNIST dataset.
'''

batch_size = 128
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28) #torch NCHW
# X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)	 #torch NCHW
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)	 #tf NHWC
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)		 #tf NHWC

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

model = Sequential()
# model.add(Convolution2D(32, 3));model.add(Activation('relu'));
# model.add(Convolution2D(32, 3, input_shape=(28, 28, 1)));model.add(Activation('relu'));
# model.add(Convolution2D(32, 3, activation='relu', input_shape=(28, 28, 1)))
from keras.layers import Input;model.add(Input(shape=(28,28,1)));
model.add(Convolution2D(32, 3));model.add(Activation('relu'))
model.add(Convolution2D(32, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

history=model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print(history.history.keys())
import matplotlib.pyplot as plt
#  "Accuracy"
plt.plot(history.history['accuracy'])#sometimes it's history.history['acc']
plt.plot(history.history['val_accuracy']) #sometimes it's history.history['val_acc']?
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



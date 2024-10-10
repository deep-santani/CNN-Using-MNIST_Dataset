import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras import backend as k
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape dataset
img_rows, img_cols = 28, 28
if k.image_data_format() == 'channels_first': 
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    inpx = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    inpx = (img_rows, img_cols, 1)

x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Model definition
input_layer = Input(shape=inpx)
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(3, 3))(x)
x = Dropout(0.5)(x)
x = Flatten()(x)
x = Dense(250, activation='sigmoid')(x)
output_layer = Dense(10, activation='softmax')(x)

model = Model(input_layer, output_layer)
model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

# Train and evaluate
model.fit(x_train, y_train, epochs=12, batch_size=500)
score = model.evaluate(x_test, y_test, verbose=0)
print('loss=', score[0], 'accuracy=', score[1])

# Single prediction
pred = model.predict(x_test)
print(np.argmax(np.round(pred[2])))

# Plot the prediction
plt.imshow(x_test[2].reshape(28, 28), cmap=plt.cm.binary)
plt.show()

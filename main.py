#! python

import tensorflow as tf 

import os


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

print(f'length of x_test {len(x_test)}')


model_filename= 'ibm_mnist_digits_2lyrs_3epochs.model'



def print_bitmap(digit):
    for i in digit:
        j = [f'{x:02X}' if x > 0 else '  ' for x in i]
        j.append('|')
        j = ''.join(j)
        print(j)
    print('--------------------------------------------------')




# print(y_train[0])
# plt.imshow(x_train[0], plt.cm.binary )
# plt.show()

original_image_test = x_test

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


if not os.path.exists(model_filename):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',  
                loss = 'sparse_categorical_crossentropy',
                metrics= 'accuracy' )
    model.fit(x_train, y_train, epochs=3)

    model.save('ibm_mnist_digits_2lyrs_3epochs.model')

else:
    model = tf.keras.models.load_model(model_filename)


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

predictions = model.predict(x_test)

import numpy as np 


for i in range(10):

    offset = 3000


    # plt.imshow(x_test[i+offset], plt.cm.binary )
    # plt.show()

    prediction = np.argmax(predictions[i+offset])
    
    print(f'PREDICTED {prediction}')
    print_bitmap(original_image_test[i+offset])

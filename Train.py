import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam,SGD
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import VGG_16

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#session = tf.Session(config=config)
#tf.config.gpu.set_per_process_memory_fraction(0.4)
seed = 124
InputPath = "data/"
X = pickle.load(open(InputPath+"X_weighted.pickle","rb"))
y = pickle.load(open(InputPath+"y_weighted.pickle","rb"))
X,y=shuffle(X,y, random_state=seed)
X = X/255.0
y = tf.keras.utils.to_categorical(y).astype(int)
print(X.shape)
print(y.shape)
print(np.amax(X))

print(np.amin(X))


train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=seed)

weights = [175,215,215,215,400]
total = sum(weights)
weights[:] = [x / total for x in weights]
print(weights)
classWeights = { i : weights[i] for i in range(0, len(weights) ) }
print(classWeights)

NAME = "VGG-{}".format(int(time.time()))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format(NAME))



model = VGG_16()
adam = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer=adam, loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])
print(model.summary())


datagen = ImageDataGenerator(
                            rotation_range=360,
                             width_shift_range=0.05,
                             height_shift_range=0.05,
                             shear_range=0.10,
                             zoom_range=0.3,
                             horizontal_flip=True, 
                             vertical_flip=True,
                             brightness_range=[0.8,1.0],
                             rescale=1./255.
                            )


checkpoint_path = "models/"+NAME+".hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/scalars/" + NAME

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

history = model.fit(datagen.flow(train_X, train_y, batch_size=16), epochs=1000,validation_data=(test_X, test_y),class_weight=classWeights,callbacks=[cp_callback,tensorboard_callback])

NAME = "VGG_withoutRegulation-{}".format(int(time.time()))

checkpoint_path = "models/"+NAME+".hdf5"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

logdir = "logs/scalars/" + NAME

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
history = model.fit(x=train_X, y=train_y, batch_size=32, epochs=1000, 
                    validation_data=(test_X, test_y),class_weight=classWeights,callbacks=[cp_callback,tensorboard_callback])


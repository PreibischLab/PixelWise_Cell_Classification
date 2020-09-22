import sys
sys.path.append('../')

from lib.train.models import unet_valid
from lib.train.data_generator import batch_data_generator

from params import *
from datetime import date
from tensorflow.keras.callbacks import ModelCheckpoint
from lib.train.callbacks import PlotLosses,SaveHistory
# , LearningRateScheduler, ReduceLROnPlateau
import os
import json

today = date.today()

d1 = today.strftime("%d_%m_%Y_")
base_name = d1+'long_run'

model_file = os.path.join(OUTPUT_PATH,base_name+'.hdf5')
print(model_file)


gen_train_fast = batch_data_generator(ZARR_FOLDER_TRAIN, batch_size=gp_batch_size
                                              ,voxel_shape = gp_voxel_shape,
                                              input_shape= gp_input_shape,
                                              output_shape = gp_output_shape,without_background=True,validate = True )

gen_test_fast = batch_data_generator(ZARR_FOLDER_TEST, batch_size=gp_batch_size
                                              ,voxel_shape = gp_voxel_shape,
                                              input_shape= gp_input_shape,
                                             output_shape = gp_output_shape,without_background = True,validate = True )


im,mask = next(gen_train_fast) 
print(im.shape)
print(mask.shape)

test_im = im[0]
test_mask = mask[0]

print('Img size: {} {} < {} | Mask size: {} {} <{}'.format(test_im.shape,test_im.dtype,test_im.max()
                                                                   ,test_mask.shape,test_mask.dtype,test_mask.max()))



# ### Model
unet_model = unet_valid(input_size = unet_input_size ,output_size = unet_output_size)
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
import tensorflow as tf

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits
# cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(targets,tf.float32))

# loss       = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1))

# prediction = tf.sigmoid(logits)
# output     = tf.cast(self.prediction > threshold, tf.int32)
# train_op   = tf.train.AdamOptimizer(0.001).minimize(loss)
# mean_squared_error


unet_model.compile(optimizer = Adam(lr = 1e-4), loss = cross_entropy , metrics=['accuracy', MeanIoU( num_classes=unet_output_size)])


unet_model.summary()


# ### Train
GRAPH_NAME = os.path.join(GRAPHS_FOLDER,base_name+'.png')
plot_losses = PlotLosses(path = GRAPH_NAME)

history2_file = os.path.join(OUTPUT_PATH,base_name+'.txt')
save_history = SaveHistory(path=history2_file)

# mean_io_u
model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss',verbose=0, save_best_only=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
#                               patience=5, min_lr=0.000001, verbose=1,  cooldown=1)



history = unet_model.fit_generator(gen_train_fast,validation_data =gen_test_fast,validation_steps=5,
                    steps_per_epoch=20,epochs=20000,callbacks=[model_checkpoint,save_history],verbose=1)


# ### Test



history_v2_dict = history.history
json.dump(str(history_v2_dict), open(history2_file, 'w'))

print(str(history_v2_dict))




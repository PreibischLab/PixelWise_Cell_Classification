import model 
import data 
import matplotlib.pyplot as plt
from datetime import date
from importlib import reload

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import os

reload(data)
reload(model)

# for Gunpowder
ZARR_FOLDER_TRAIN = '../../../../data/cells/raw/zarr/train/'
ZARR_FOLDER_TEST = '../../../../data/cells/raw/zarr/test/'
gp_batch_size= 12
gp_voxel_shape = [1,1,1]
gp_input_shape= [256, 256,4]
gp_output_shape = [ 256, 256,4]

# For unet
OUTPUT_PATH = '../../../../data/cells/models/'

unet_input_size = (256,256,4)
unet_output_size = 3


today = date.today()

d1 = today.strftime("%d_%m_%Y_")
model_name = d1+'unet_without_bg_cells.hdf5'
model_name


# ### Check data 


gen_train_fast = data.generate_fast_training_batch_without_background(ZARR_FOLDER_TRAIN, batch_size=gp_batch_size
                                              ,voxel_shape = gp_voxel_shape,
                                              input_shape= gp_input_shape,output_shape = gp_output_shape )
gen_test_fast = data.generate_fast_training_batch_without_background(ZARR_FOLDER_TEST, batch_size=gp_batch_size
                                              ,voxel_shape = gp_voxel_shape,
                                              input_shape= gp_input_shape,output_shape = gp_output_shape )


im,mask = next(gen_train_fast)
print(im.shape)


test_im = im[0]
test_mask = mask[0]

print('Img size: {} {} < {} | Mask size: {} {} <{}'.format(test_im.shape,test_im.dtype,test_im.max()
                                                                   ,test_mask.shape,test_mask.dtype,test_mask.max()))


# ### Model


unet_model = model.unet(input_size = unet_input_size ,output_size = unet_output_size)


unet_model.summary()




# ### Train


model_file = os.path.join(OUTPUT_PATH,model_name)
# mean_io_u
model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss',verbose=1, save_best_only=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
#                               patience=5, min_lr=0.000001, verbose=1,  cooldown=1)


# In[11]:


history = unet_model.fit_generator(gen_train_fast,validation_data =gen_test_fast,validation_steps=10,steps_per_epoch=100,epochs=20000,callbacks=[model_checkpoint])


# ### Test
import json
history2_file = os.path.join(OUTPUT_PATH,d1+'history_unet_without2_background_cells.json')
history_v2_dict = history.history
json.dump(str(history_v2_dict), open(history2_file, 'w'))







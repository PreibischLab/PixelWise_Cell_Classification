#!/usr/bin/env python
# coding: utf-8



import model 
import data 
import matplotlib.pyplot as plt
# from importlib import reload
# reload(data)
# reload(model)



# for Gunpowder
ZARR_FOLDER_TRAIN = '../../../../data/cells/raw/zarr/train/'
ZARR_FOLDER_TEST = '../../../../data/cells/raw/zarr/test/'
gp_batch_size= 2
gp_voxel_shape = [1,1,1]
gp_input_shape= [572, 572,4]
gp_output_shape = [ 388, 388,3]

# For unet
OUTPUT_PATH = '../../../../data/cells/models/'

unet_input_size = (572, 572,4)
unet_output_size = 3




from datetime import date

today = date.today()

d1 = today.strftime("%d_%m_%Y_")
base_name = d1+'unet_without_bg_cells_valid'
model_name = base_name+'.hdf5'
model_name


gen_train_fast = data.generate_fast_training_batch_different_shape(ZARR_FOLDER_TRAIN, batch_size=gp_batch_size
                                              ,voxel_shape = gp_voxel_shape,
                                              input_shape= gp_input_shape,output_shape = gp_output_shape )
gen_test_fast = data.generate_fast_training_batch_different_shape(ZARR_FOLDER_TEST, batch_size=gp_batch_size
                                              ,voxel_shape = gp_voxel_shape,
                                              input_shape= gp_input_shape,output_shape = gp_output_shape )


im,mask = next(gen_train_fast)
print(im.shape)
print(mask.shape)




test_im = im[0]
test_mask = mask[0]

print('Img size: {} {} < {} | Mask size: {} {} <{}'.format(test_im.shape,test_im.dtype,test_im.max()
                                                                   ,test_mask.shape,test_mask.dtype,test_mask.max()))



# ### Model



unet_model = model.unet_valid(input_size = unet_input_size ,output_size = unet_output_size)


# In[9]:


unet_model.summary()




# ### Train



import os
GRAPHS_FOLDER = '../../../../data/cells/graphs/'
GRAPH_NAME = os.path.join(GRAPHS_FOLDER,base_name+'.png')

from tensorflow import keras
# import IPython.display as display
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.history = {}
        self.fig = plt.figure()
        
        self.logs = []
#         plt.show()

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        for l in logs:
            if l not in self.history:
                self.history[l]=[]
            self.history[l].append(logs.get(l))
        self.i += 1
        
#         display.clear_output(wait=True)
        for l in logs:
            plt.plot(self.x, self.history[l], label=l)
        plt.legend()
        plt.savefig(GRAPH_NAME)
        # plt.show();
        
plot_losses = PlotLosses()



from tensorflow.keras.callbacks import ModelCheckpoint
# , LearningRateScheduler, ReduceLROnPlateau
import os

model_file = os.path.join(OUTPUT_PATH,model_name)
# mean_io_u
model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss',verbose=0, save_best_only=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
#                               patience=5, min_lr=0.000001, verbose=1,  cooldown=1)



history = unet_model.fit_generator(gen_train_fast,validation_data =gen_test_fast,validation_steps=5,steps_per_epoch=30,epochs=1000,callbacks=[model_checkpoint,plot_losses],verbose=1)


# ### Test


import json
history2_file = os.path.join(OUTPUT_PATH,base_name+'.json')
history_v2_dict = history.history
json.dump(str(history_v2_dict), open(history2_file, 'w'))


# In[ ]:


print(str(history_v2_dict))




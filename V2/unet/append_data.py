import zarr
import os
import numpy as np


def add_data(zarr_file,name,img_shape,boxes_positions,data):
    store = zarr.DirectoryStore(zarr_file)
    root = zarr.group(store=store, overwrite=False)
    pred_data = root.create_dataset(name, data=np.zeros((1024, 1024, 4),dtype=np.float32), chunks=(256, 256,1))
    for bp, d in zip(boxes_positions,data):
        pred_data[bp[0]:bp[2],bp[1]:bp[3],:] = d

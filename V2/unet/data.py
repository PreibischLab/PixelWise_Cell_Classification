import gunpowder as gp
import zarr
import matplotlib.pyplot as plt
import os
import numpy as np


def generate_fast_training_batch(input_path, batch_size=12,voxel_shape = [1,1,1],input_shape= [256, 256,4 ],output_shape = [ 256, 256,4] ):
    raw = gp.ArrayKey('raw')
    gt = gp.ArrayKey('ground_truth')
    files = os.listdir(input_path)
    files = [os.path.join(input_path,f) for f in files ]
#     print(files)
    pipeline =( tuple (
        gp.ZarrSource(
            files[t],  # the zarr container
            {raw: 'raw', gt : 'ground_truth'},  # which dataset to associate to the array key
            {raw: gp.ArraySpec(interpolatable=True,dtype=np.dtype('float32'),voxel_size=voxel_shape),
             gt: gp.ArraySpec(interpolatable=True,dtype=np.dtype('float32'),voxel_size=voxel_shape)}  # meta-information
        )
        + gp.RandomLocation()
        for t in range(len(files))
    )
               + gp.RandomProvider()
               +gp.Stack(batch_size)
              )
#     voxel_size = gp.Coordinate(voxel_shape)
#     *voxel_size
    input_size = gp.Coordinate(input_shape)
    output_size = gp.Coordinate(output_shape)
    
    request = gp.BatchRequest()
    request.add(raw,input_size)
    request.add(gt,output_size)
    
    while 1:
        with gp.build(pipeline):
            batch = pipeline.request_batch(request)
            yield batch[raw].data, batch[gt].data
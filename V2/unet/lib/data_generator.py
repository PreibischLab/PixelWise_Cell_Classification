import gunpowder as gp
import zarr
import matplotlib.pyplot as plt
import os
import numpy as np
import math
            
def generate_fast_training_batch(input_path, batch_size=12,voxel_shape = [1,1,1],
                                 input_shape= [240, 240,4],output_shape = [240, 240,4] ):
    raw = gp.ArrayKey('raw')
    gt = gp.ArrayKey('ground_truth')
    files = os.listdir(input_path)
    files = [os.path.join(input_path,f) for f in files ]
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

    input_size = gp.Coordinate(input_shape)
    output_size = gp.Coordinate(output_shape)
    
    request = gp.BatchRequest()
    request.add(raw,input_size)
    request.add(gt,output_size)
    
    while 1:
        with gp.build(pipeline):
            batch = pipeline.request_batch(request)
            yield batch[raw].data, batch[gt].data
          

def generate_fast_training_batch_without_background(input_path, batch_size=12,voxel_shape = [1,1,1],
                                 input_shape= [240, 240,4],output_shape = [240, 240,4] ):
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
            yield batch[raw].data, batch[gt].data[:,:,:,1:4]
            
def generate_fast_training_batch_different_shape(input_path, batch_size=12,voxel_shape = [1,1,1],
                                 input_shape= [240, 240,4],output_shape = [240, 240,4] ):
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
    request.add(gt,input_size)
    diff = input_shape[1] - output_shape[1]
    diff = int(diff/2)
    max_p = input_shape[1]-diff
    print('Difference padding: {}'.format(diff))
    while 1:
        with gp.build(pipeline):
            batch = pipeline.request_batch(request)
            im = batch[raw].data
            out = batch[gt].data
            out = out[:,diff:max_p,diff:max_p,1:4]
            yield im, out.argmax(axis=3)
            
def generate_test_batch(file,blocks, voxel_shape = [1,1,1]):
    raw = gp.ArrayKey('raw')
    gt = gp.ArrayKey('ground_truth')
    instances = gp.ArrayKey('instances')
    
#     print(files)
    pipeline = gp.ZarrSource(
            file,  # the zarr container
            {raw: 'raw', gt : 'ground_truth',instances : 'instances'},  # which dataset to associate to the array key
            {raw: gp.ArraySpec(interpolatable=True,dtype=np.dtype('float32'),voxel_size=voxel_shape),
             gt: gp.ArraySpec(interpolatable=True,dtype=np.dtype('float32'),voxel_size=voxel_shape)
             ,instances: gp.ArraySpec(interpolatable=False,dtype=np.dtype('int32'),voxel_size=[1,1])
            }  # meta-information
    )
    
              
#     voxel_size = gp.Coordinate(voxel_shape)
#     *voxel_size
#     input_size = gp.Coordinate(input_shape)
#     gt_size = gp.Coordinate(gt_shape)
#     mask_size = gp.Coordinate(mask_shape)
    
    request = gp.BatchRequest()
#     request.add(raw,input_size)
#     request.add(gt,gt_size)
# #     request.add(mask,mask_size)
#     request[mask] = gp.Roi((0, 0,0), (240, 240,3))
    result = {'raw': [] , 'instances': [] , 'gt' : [] }
    with gp.build(pipeline):
        for b in blocks:
            print('Processing block {}'.format(b))
            request[raw] = gp.Roi((b[0], b[1],0), (256,256,4))
            request[gt] = gp.Roi((b[0], b[1],0), (256,256,4))
            request[instances] = gp.Roi((b[0], b[1]), (256,256))
            batch = pipeline.request_batch(request)
            result['raw'].append(batch[raw].data)
            result['gt'].append(batch[gt].data)
            result['instances'].append(batch[instances].data)
    return result

def generate_raw_batch(file,blocks, voxel_shape = [1,1,1]):
    raw = gp.ArrayKey('raw')

    pipeline = gp.ZarrSource(
            file,  # the zarr container
            {raw: 'raw'},  # which dataset to associate to the array key
            {raw: gp.ArraySpec(interpolatable=True,dtype=np.dtype('float32'),voxel_size=voxel_shape)
            }  # meta-information
    )
    
    request = gp.BatchRequest()
    result = []
    with gp.build(pipeline):
        for b in blocks:
            print('Processing block {}'.format(b))
            request[raw] = gp.Roi((b[0], b[1],0), (256,256,4))
            batch = pipeline.request_batch(request)
            result.append(batch[raw].data)
    return result

def add_data(zarr_file,name,img_shape,boxes_positions,data):
    store = zarr.DirectoryStore(zarr_file)
    root = zarr.group(store=store, overwrite=False)
    pred_data = root.create_dataset(name, data=np.zeros((1024, 1024, 4),dtype=np.float32), chunks=(256, 256,1))
#     root = zarr.open(zarr_file, mode='w')
#     pred_data = root.create_dataset(name, data=np.zeros(img_shape,dtype=np.float32), chunks=(256, 256,1))
    for bp, d in zip(boxes_positions,data):
        pred_data[bp[0]:bp[2],bp[1]:bp[3],:] = d
#     root.close()
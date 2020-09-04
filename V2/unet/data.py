import gunpowder as gp
import zarr
import matplotlib.pyplot as plt
import os
import numpy as np


def generate_fast_training_batch(input_path, batch_size=12,voxel_shape = [1,1,1],
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
            yield batch[raw].data, batch[gt].data
          
        
def generate_test_batch(file,blocks, voxel_shape = [1,1,1]):
    raw = gp.ArrayKey('raw')
    gt = gp.ArrayKey('ground_truth')
    mask = gp.ArrayKey('mask')
    
#     print(files)
    pipeline = gp.ZarrSource(
            file,  # the zarr container
            {raw: 'raw', gt : 'ground_truth',mask : 'mask'},  # which dataset to associate to the array key
            {raw: gp.ArraySpec(interpolatable=True,dtype=np.dtype('float32'),voxel_size=voxel_shape),
             gt: gp.ArraySpec(interpolatable=True,dtype=np.dtype('float32'),voxel_size=voxel_shape),
            mask: gp.ArraySpec(interpolatable=True,dtype=np.dtype('int32'),voxel_size=voxel_shape)
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
    result = {'raw': [] , 'mask': [] , 'gt' : [] }
    with gp.build(pipeline):
        for b in blocks:
            print('Processing block {}'.format(b))
            request[raw] = gp.Roi((b[0], b[1],0), (256,256,4))
            request[gt] = gp.Roi((b[0], b[1],0), (256,256,4))
            request[mask] = gp.Roi((b[0], b[1],0), (256,256,3))
        
            batch = pipeline.request_batch(request)
            result['raw'].append(batch[raw].data)
            result['gt'].append(batch[gt].data)
            result['mask'].append(batch[mask].data)
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
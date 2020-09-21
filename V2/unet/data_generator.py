import gunpowder as gp
import zarr
import os
import numpy as np
            
def batch_data_generator(input_path,batch_size=12,voxel_shape = [1,1,1],
                             input_shape= [240, 240,4],
                             output_shape = [240, 240,4],
                             without_background = False,
                                 mix_output = False):
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
    input_size = gp.Coordinate(input_shape)
    output_size = gp.Coordinate(output_shape)

    request = gp.BatchRequest()
    request.add(raw,input_size)
    request.add(gt,input_size)
    diff = input_shape[1] - output_shape[1]
    diff = int(diff/2)
    max_p = input_shape[1]-diff
    different_shape = diff > 0
    if different_shape:
        print('Difference padding: {}'.format(diff))
    while 1:
        with gp.build(pipeline):
            batch = pipeline.request_batch(request)
            im = batch[raw].data
            out = batch[gt].data
            if different_shape:
                out = out[:,diff:max_p,diff:max_p,:]
            if without_background:
                out = out[:,:,:,1:4]
            if mix_output:
                out = out.argmax(axis=3).astype(float)
            yield im, out
            
# def generate_test_batch(file,blocks, voxel_shape = [1,1,1]):
#     raw = gp.ArrayKey('raw')
#     gt = gp.ArrayKey('ground_truth')
#     instances = gp.ArrayKey('instances')
#     pipeline = gp.ZarrSource(
#             file, 
#             {raw: 'raw', gt : 'ground_truth',instances : 'instances'},  
#             {raw: gp.ArraySpec(interpolatable=True,dtype=np.dtype('float32'),voxel_size=voxel_shape),
#              gt: gp.ArraySpec(interpolatable=True,dtype=np.dtype('float32'),voxel_size=voxel_shape)
#              ,instances: gp.ArraySpec(interpolatable=False,dtype=np.dtype('int32'),voxel_size=[1,1])
#             }  
#     )
   
#     request = gp.BatchRequest()

#     result = {'raw': [] , 'instances': [] , 'gt' : [] }
#     with gp.build(pipeline):
#         for b in blocks:
#             print('Processing block {}'.format(b))
#             request[raw] = gp.Roi((b[0], b[1],0), (256,256,4))
#             request[gt] = gp.Roi((b[0], b[1],0), (256,256,4))
#             request[instances] = gp.Roi((b[0], b[1]), (256,256))
#             batch = pipeline.request_batch(request)
#             result['raw'].append(batch[raw].data)
#             result['gt'].append(batch[gt].data)
#             result['instances'].append(batch[instances].data)
#     return result

# def generate_raw_batch(file,blocks, voxel_shape = [1,1,1]):
#     raw = gp.ArrayKey('raw')

#     pipeline = gp.ZarrSource(
#             file,  # the zarr container
#             {raw: 'raw'},  # which dataset to associate to the array key
#             {raw: gp.ArraySpec(interpolatable=True,dtype=np.dtype('float32'),voxel_size=voxel_shape)
#             }  # meta-information
#     )
    
#     request = gp.BatchRequest()
#     result = []
#     with gp.build(pipeline):
#         for b in blocks:
#             print('Processing block {}'.format(b))
#             request[raw] = gp.Roi((b[0], b[1],0), (256,256,4))
#             batch = pipeline.request_batch(request)
#             result.append(batch[raw].data)
#     return result


import gunpowder as gp
import zarr
import os
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage            

def batch_data_aug_generator(input_path,batch_size=12,voxel_shape = [1,1,1],
                             input_shape= [240, 240,4],
                             output_shape = [240, 240,4],
                             without_background = False,
                                 mix_output = False, 
                                 validate = False,
                                 aug = seq ):
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
            #    +gp.Stack(batch_size)
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
    with gp.build(pipeline):
        while 1:
            b = 0
            imgs = []
            masks = []
            while b<batch_size:
                valid = False
                batch = pipeline.request_batch(request)
                if validate:
                    valid = validate_mask(batch[gt].data)
                else:
                    valid = True
                
                while(valid == False):
                    batch = pipeline.request_batch(request)
                    valid = validate_mask(batch[gt].data)
                im = batch[raw].data
                out = batch[gt].data
                # im,out = augmentation(im,out,seq)
                if different_shape:
                    out = out[diff:max_p,diff:max_p,:]
                if without_background:
                    out = out[:,:,1:4]
                if mix_output:
                    out = out.argmax(axis=3).astype(float)
                imgs.append(im)
                masks.append(out)
                b = b+1
            yield augmentation(np.asarray(imgs), np.asarray(masks),seq)


def augmentation(images, masks,seq):
    pos = images.shape[3]
    all_stacks = np.concatenate((images, masks), axis=3)
    images_aug = seq(images=all_stacks)
    return images_aug[:,:,:,:pos],  images_aug[:,:,:,pos:]

    

# Check if all the layers are > 0
def validate_mask(mask):
    sh = mask.shape
    if(len(sh) != 3):
        print("invalid shape!")
        return False
    # if(sh[0]>1):
    #     print("n=more than one batch not implemented yet")
    #     return False
    for i in range(sh[2]):
        if mask[:,:,i].max() < 1:
            return False
    return True

import os
from skimage import io
import numpy as np
import pandas as pd
import zarr

RAW = 'raw'
INSTANCES = 'instances'
GT = 'ground_truth'

def get_base_name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def get_files(folder):
    files = os.listdir(folder)
    files = [os.path.join(folder,f) for f in files]
    return files

def get_right_mask(masks,base_name):
    result = []
    for m in masks:
        if base_name in m :
            result.append(m)
    result.sort()
    return result[0]

def generate_pairs(csv_folder,mask_folder):
    csvs = get_files(csv_folder)
    masks = get_files(mask_folder)
    pairs = {}
    for f in csvs:
        if f.endswith('.csv'):
            base_name = get_base_name(f)
            mask = get_right_mask(masks,get_base_name(f))
            pairs[f]=mask
    return pairs

def read_img(path):
    print('Reading image: {}'.format(path))
    im = io.imread(path)
    return im


def generateCategoryImage(path_instance_image,path_csv):
    data = pd.read_csv(path_csv) 
    img = read_img(path_instance_image)
    result = np.zeros_like(img)
    for i,val,dims in zip(data['annotationId'],data['annotationVal'],data['spaceDims']):
        result = result + (img==i)*val
    return result

def generateCategoryImagefromList(path_instance_image,list_inst):
#     data = pd.read_csv(path_csv) 
    img = read_img(path_instance_image)
    result = np.zeros_like(img)
    for i,val in enumerate(list_inst):
        result = result + (img==(i+1))*val
    return result

def normalize_mask(mask,list_vals):
    size = list_vals.size
    shape = (mask.shape+tuple([size]))
    result = np.zeros(shape, dtype=np.float32)
    for i in range(size):
        tmp = np.zeros_like(mask)
        tmp [mask == list_vals[i]] = 1.0
        result[:,:,i] = tmp
    return result

def normalize_image(img):
    img = img.astype(np.float32)
    img = img / 255.
    return img


def create_zarr_per_pair(input_folder,instances_folder,csv_folder,zarr_folder, values = [0,1,2,3], normalize = False):

    if not os.path.exists(zarr_folder):
        os.makedirs(zarr_folder)
    pairs = generate_pairs(csv_folder,instances_folder)
    size = len(pairs)
    for i,(csv_path, inst_path) in enumerate(pairs.items()):
        base_name = get_base_name(csv_path)
        zarr_file = os.path.join(zarr_folder,base_name+'.zarr') 
        print('{} - {}'.format(i,zarr_file))
        root = zarr.open(zarr_file, mode='w')
        
        input_file = os.path.join(input_folder,base_name+'.tif')   
        im_input = read_img(input_file)
        im_inst = read_img(inst_path)
        
        im_categ = generateCategoryImage(inst_path,csv_path)
        
        # removed the last value because it is the error category
#         categ_values = np.unique(im_categ)[:4]
        categ_values = np.array(values)
        shape_input = im_input.shape
        shape_categ = (im_inst.shape+tuple([categ_values.size]))
        shape_instances =  im_inst.shape
        print('The value to be activated in mask: {} '.format(categ_values))
        
        print('Img size        : {} {} < {}'.format(shape_input,im_input.dtype,im_input.max()))
        print('Instances size  : {} {} < {}'.format(shape_instances,im_inst.dtype,im_inst.max()))
        print('Categories size : {} {} < {}'.format(shape_categ,im_categ.dtype,im_categ.max()))
        input_dtype = 'f8' if normalize else 'i8'
        root.zeros(RAW, shape=shape_input, chunks=(256, 256,1), dtype='i8')
        root.zeros(GT, shape=shape_categ, chunks=(256, 256,1), dtype='f8')
#         root.zeros(PREDICTION, shape=shape_mask, chunks=(256, 256,1), dtype='f8')
        root.zeros(INSTANCES, shape=shape_instances, chunks=(256, 256), dtype='i8')
        if normalize:
            im_input = normalize_image(im_input)
        else:
            im_input = im_input.astype(np.float32)
        normalized_mask = normalize_mask(im_categ,categ_values)
        print('After normalization: ')
        print('Img size        : {} {} < {}'.format(im_input.shape,im_input.dtype,im_input.max()))
        print('Categories size : {} {} < {}'.format(normalized_mask.shape,normalized_mask.dtype,normalized_mask.max()))
        root[RAW] = im_input
        root[GT] = normalized_mask
        root[INSTANCES] = im_inst
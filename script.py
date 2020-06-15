import numpy as np
import time, os, sys
import mxnet as mx
import matplotlib.pyplot as plt
import glob
import sys
import pandas as pd
from cellpose import models, utils
from cellpose import plot, transforms
import cv2

print('hello')

CSV_FOLDER = 'data/raw/CSV_patryk/'
TIFF_FOLDER = 'data/raw/RAW_TIFF/'
TEST_IMAGE = 'data/raw/RAW_TIFF/20072019_ND9_ND11_DIV0-1_Daam1_aTub_Phallo_10_ch_4.tif'
TEST_CSV = 'data/raw/CSV_patryk/20072019_ND9_ND11_DIV0-1_Daam1_aTub_Phallo_10_ch_4.csv'
BLOCK_ANNOTATION = 'data/raw/block_annotation2.csv'
BLOCK_SIZE = 50
TO_IGNORE = 'ND8_DIV0+2h_'


use_gpu = utils.use_gpu()
if use_gpu:
    device = mx.gpu()
else:
    device = mx.cpu()



def processAll(folder_csv,folder_tiff):
    X = []
    y = []
    all_csv,all_tiff = prepareList(folder_csv,folder_tiff,TO_IGNORE)
#     all_csv= all_csv[:2]
#     all_tiff= all_tiff[:2]
    annotation = pd.read_csv(BLOCK_ANNOTATION, header=None)
    all_mask = getMasks(all_tiff)
    save(all_mask,all_tiff)
    for csv, tiff, mask in zip(all_csv, all_tiff,all_mask):
        
        Xt, yt = processImage(csv,tiff,mask,annotation)
        X = X + Xt
        y = y + yt
    return X,y
 
def save(all_mask,all_tiff):
    for i in range(len(all_tiff)):
        name = os.path.basename(all_tiff[0]).replace('.tif','.jpg')
        cv2.imwrite('data/masks/%s'%name, all_mask[i])

def processImage(csv,tiff,mask,annotations):
    print('Processing %s'%os.path.basename(tiff))
    Xt = []
    yt = []
    data = pd.read_csv(csv)
    size = data.shape[0]
    vals = getVals(data,mask)
    image = plt.imread(tiff)
    base_name = os.path.basename(tiff).split('.')[0].replace("_ch_4", "")
    for i in range(size):
        x,y = getPoint(data,i)
        val = vals[i]
        block_name = '%s_%d'%(base_name,i)
        annotation = getAnnotation(annotations,block_name)
        if annotation == -1 :
            continue
        if val > 0 :
            activated_mask = activate(mask,val)
            activated_mask = np.stack((activated_mask,)*3, axis=-1)
            masked_image = activateImage(image, activated_mask)
            height, width = image.shape
            if(checkPosition(height, width,y,x,BLOCK_SIZE)==False):
                continue
            cropped_image = cropImage(masked_image,x,y,BLOCK_SIZE)
            Xt.append(cropped_image)
            yt.append(annotation)
    print('%s created %d images'%(os.path.basename(tiff),len(Xt)))
    return Xt, yt

def getAnnotation(annotations, name):
    for file,val in zip(annotations[0],annotations[1]):
        if name in file:
            return val
    return -1
        
def checkPosition(height,width,y,x,size):
    if x<size :
        return False
    if y<size :
        return False
    if x+size>width :
        return False
    if y+size>height:
        return False
    return True

def cropImage(image,x,y,size):
    height, width = image.shape
    x1 = max([0,x-size+1])
    y1 = max([0,y-size+1])
    x2 = min([width,x+size])
    y2 = min([height,y+size])
#     base=np.zeros((size*2,size*2),dtype=np.uint8)
    return image[y1:y2, x1:x2]
    
def show(img,title):
    fig2 = plt.figure(figsize = (15,15))
    ax3 = fig2.add_subplot(111)
    ax3.imshow(img, interpolation='none')
    ax3.set_title(title)

def getPoint(data,i):
    return int(data['x'][i]),int(data['y'][i])

def activate(array,value):
    return (array == value) * 1

def activateImage(image,mask):
    return (mask == 1)*image

def getVals(data,mask):
    vals = []
    not_found = 0
    duplicate = 0
    for i in range(data.shape[0]):
        x,y = getPoint(data,i)
        val = masks[0][y][x]
        if val == 0 :
            not_found = not_found +1
            print("%d- %d Not found!"%(i,val))
        else:
            if val in vals:
                duplicate = duplicate +1
                print("%d- %d already exists!"%(i,val))
        vals.append(val)
    print("not found: %d"%not_found)
    print("duplicate: %d"%duplicate)
    return vals,not_found,duplicate

    
def prepareList(folder_csv,folder_tiff,to_ignore):
    all_csv = []
    all_tiff = []
    list_tiff = os.listdir(folder_tiff)
    
    for tiff in list_tiff:
        tiff_path = os.path.join(folder_tiff,tiff)
        csv = "%s.csv"%tiff.split('.')[0]
        csv_path = os.path.join(folder_csv,csv)
        if not os.path.exists(tiff_path):
            print('%s not exist'%tiff)
            continue
        if not os.path.exists(csv_path):
            print('%s not exist'%csv)
            continue
        if to_ignore in tiff :
            print('%s to ignore'%csv)
            continue
        all_csv.append(csv_path)
        all_tiff.append(tiff_path)
    return all_csv,all_tiff

def getMasks(list_tiff):
    print('Detecting masks ..')
    model = models.Cellpose(device, model_type='cyto')
    imgs = [plt.imread(f) for f in list_tiff]
    nimg = len(imgs)
    channel =[2,1]
    channels = [channel for i in range(len(list_tiff))]
    masks, _, _, _ = model.eval(imgs, rescale=None, channels=channels)
    return masks


def main():
    X,y = processAll(CSV_FOLDER,TIFF_FOLDER)
    print(X.shape)
    print(y.shape)



    X = np.array(X).reshape(-1,100,100,3)
    y = np.array(y).reshape(-1,1)



    pickle_out = open("data/X2_masked.pickle","wb")
    pickle.dump(X,pickle_out)
    pickle_out.close()

    pickle_out = open("data/y2_masked.pickle","wb")
    pickle.dump(y,pickle_out)
    pickle_out.close()

    print(X.shape) 
    print(y.shape)

if __name__ == "__main__":
    main()




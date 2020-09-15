import numpy as np
import time
from tqdm import tqdm



def mix_channels(img):
    result = np.zeros(img.shape[:2])
    for i in range(img.shape[2]):
        result = result+img[:,:,i]*(i+1)
    return result

def get_prob_categories_per_instance(instances_img, categories_img,instance):
    start_time = time.time()
    categories = {}
    shape = instances_img.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if instances_img[i][j]==instance:
                cat = categories_img[i][j]
                if cat in categories:
                    categories[cat]=categories[cat]+1
                else:
                    categories[cat] = 0
    print("Time: %s seconds" % (time.time() - start_time))
    return categories


def get_prob_categories_per_instance_v2(instances_img, categories_img,instance,test=True):
    start_time = time.time()
    current_image = np.zeros(instances_img.shape)
    current_image[instances_img==instance]=1
    # +1 is to get the count of background in that instance
    current_image = current_image*(categories_img+1)
    unique_elements, counts_elements = np.unique(current_image, return_counts=True)
    unique_elements = unique_elements[1:]
    counts_elements = counts_elements[1:]
    sum_count = np.sum(counts_elements)
    categories = dict(zip((unique_elements.astype(np.int)-1), counts_elements/sum_count))
    if test :
        print("Time: %s seconds" % (time.time() - start_time))
    return categories

def getMax(categories):
    max_val = max(categories.values())
    for cat in categories:
        if max_val == categories[cat]:
            return cat,max_val
 
    


def get_categories_map_per_instances(instances_img, categories_img,gt=None,threshold=0.7):
    cat_per_inst = []
    not_found = 0
    np_instances = instances_img.max()
    for i in range(1,np_instances+1):
        categories = get_prob_categories_per_instance_v2(instances_img,categories_img,i,test=False)
        if len(categories)== 0 :
            cat_per_inst.append(-1)
            continue
        k,v = getMax(categories)
        if v>=threshold:
            cat_per_inst.append(k)
        else:
            cat_per_inst.append(-1)
            if gt.any() == None :
                print('{}- Not found: {}'.format(not_found,categories) )
            else:
                categories_gt = get_prob_categories_per_instance_v2(instances_img,gt,i,test=False)
                right_category,_ = getMax(categories_gt)
                print('{:.2f} - {}: {} '.format(v,k,right_category) )
                      
    return cat_per_inst
            
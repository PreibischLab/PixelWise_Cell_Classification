import numpy as np
import time



def mix_channels(img):
    result = np.zeros(img.shape[:2])
    for i in range(img.shape[2]):
        result = result+img[:,:,i]*(i+1)
    return result

def get_prob_categories_per_imstance(instances_img, categories_img,instance):
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


def get_prob_categories_per_imstance_v2(instances_img, categories_img,instance):
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
    print("Time: %s seconds" % (time.time() - start_time))
    return categories
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

def getImgAug():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([

                iaa.Fliplr(0.5),

                iaa.Flipud(0.5),

                iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)

                sometimes(iaa.MotionBlur(k=5)),

                sometimes(iaa.Affine(

                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis

                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)

                rotate=(-45, 45), # rotate by -45 to +45 degrees

                shear=(-16, 16), # shear by -16 to +16 degrees

                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)

                cval=(0, 0), # if mode is constant, use a cval between 0 and 255

                mode='constant' # use any of scikit-image's warping modes (see 2nd image from the top for examples)

            )),

            iaa.ElasticTransformation(alpha=(0, 1.0), sigma=0.8),

            ], random_order=False)
    return seq
# images_aug = seq(images=vis)
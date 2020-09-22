# Preparation
CSV_FOLDER = '/tf/data/cells/raw/csv/'
INSTANCES_FOLDER = '/tf/data/cells/raw/masks/'
INPUT_FOLDER = '/tf/data/cells/raw/input/'

RAW = 'raw'
INSTANCES = 'instances'
GT = 'ground_truth'



# for Gunpowder
ZARR_FOLDER_TRAIN = '/tf/data/cells/raw/zarr/train/'
ZARR_FOLDER_TEST = '/tf/data/cells/raw/zarr/test/'
gp_batch_size= 2
gp_voxel_shape = [1,1,1]
gp_input_shape= [572, 572,4]
gp_output_shape = [ 388, 388,3]

# For unet
OUTPUT_PATH = '/tf/data/cells/models/'

unet_input_size = (572, 572,4)
unet_output_size = 3

GRAPHS_FOLDER = '/tf/data/cells/graphs/'
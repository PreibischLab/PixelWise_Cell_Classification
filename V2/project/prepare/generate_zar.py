import sys
sys.path.append('../')
from params import *
from lib.prepare.generator import create_zarr_per_pair

ZARR_FOLDER = '/tf/data/cells/raw/zarr_normal/'
create_zarr_per_pair(INPUT_FOLDER,INSTANCES_FOLDER,CSV_FOLDER,ZARR_FOLDER,normalize=False)
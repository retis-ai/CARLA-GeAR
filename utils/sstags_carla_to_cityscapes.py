"""
Script to map the ground truth semantic segmentation tags returned by Carla to tags from the CityScapes dataset.
Usage: python sstags_carla_to_cityscapes.py --in path/to/input/folder --out path/to/output/folder

Author: Saasha Nair
"""

import os
import sys

import argparse
import imageio
import glob
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


CITYSCAPES_LABEL_TO_COLOR_AND_ID = {
    'unlabeled'            : [[  0,  0,  0], 0 ],
    'ego vehicle'          : [[  0,  0,  0], 0 ], # original = 1
    'rectification border' : [[  0,  0,  0], 0 ], # original = 2
    'out of roi'           : [[  0,  0,  0], 0 ], # original = 3
    'static'               : [[  0,  0,  0], 0 ], # original = 4
    'dynamic'              : [[111, 74,  0], 5 ],
    'ground'               : [[ 81,  0, 81], 6 ],
    'road'                 : [[128, 64,128], 7 ],
    'sidewalk'             : [[244, 35,232], 8 ],
    'parking'              : [[250,170,160], 9 ],
    'rail track'           : [[230,150,140], 10],
    'building'             : [[ 70, 70, 70], 11],
    'wall'                 : [[102,102,156], 12],
    'fence'                : [[190,153,153], 13],
    'guard rail'           : [[180,165,180], 14],
    'bridge'               : [[150,100,100], 15],
    'tunnel'               : [[150,120, 90], 16],
    'pole'                 : [[153,153,153], 17],
    'polegroup'            : [[153,153,153], 17], # original = 18
    'traffic light'        : [[250,170, 30], 19],
    'traffic sign'         : [[220,220,  0], 20],
    'vegetation'           : [[107,142, 35], 21],
    'terrain'              : [[152,251,152], 22],
    'sky'                  : [[ 70,130,180], 23],
    'person'               : [[220, 20, 60], 24],
    'rider'                : [[255,  0,  0], 25],
    'car'                  : [[  0,  0,142], 26],
    'truck'                : [[  0,  0, 70], 27],
    'bus'                  : [[  0, 60,100], 28],
    'caravan'              : [[  0,  0, 90], 29],
    'trailer'              : [[  0,  0,110], 30],
    'train'                : [[  0, 80,100], 31],
    'motorcycle'           : [[  0,  0,230], 32],
    'bicycle'              : [[119, 11, 32], 33],
    'license plate'        : [[  0,  0,142], 26 ], # original = -1
}

 
CARLA_COLOR_CITYSCAPES_LABEL = {
    (0, 0, 0): 'unlabeled', # unlabeled
    (70, 70, 70): 'building', # building
    (100, 40,40): 'fence', # fence
    (55, 90, 80): 'unlabeled', # other
    (220, 20, 60): 'person', # pedestrians
    (153, 153, 153): 'pole', # pole
    (157, 234, 50): 'road', # roadline
    (128, 64, 128): 'road', # road
    (244, 35, 232): 'sidewalk', # sidewalk
    (107, 142, 35): 'vegetation', # vegetation
    (0, 0, 142): 'car', # vehicles
    (102, 102, 156): 'wall', # wall
    (220, 220, 0): 'traffic sign', # trafficsign
    (70, 130, 180): 'sky', # sky
    (81, 0, 81): 'ground', # ground
    (150, 100, 100): 'bridge', # bridge
    (230, 150, 140): 'rail track', #railtrack
    (180, 165, 180): 'guard rail', # guardrail
    (250, 170, 30): 'traffic light', # trafficlight
    (110, 190, 160): 'static', # static
    (170, 120, 50): 'dynamic', # dynamic
    (45, 60, 150): 'unlabeled', # water
    (145, 170, 100): 'terrain', # terrain
}

def convert_single_image_to_cityscapes(image_to_map):
    """
    Function that creates a color image of Cityscapes tags and grayscale image of cityscapes labelIds for a given color image of Carla tags

    Parameters
    ---
    image_to_map: imageio.core.util.Array
        RGB image obtained from Carla's Semantic Segmentation Camera

    Returns
    ---
    Two numpy arrays, namely cityscapes_color_image and cityscapes_label_image. 
    cityscapes_label_image has the same shape as image_to_map and contains the RGB equivalents from CityScapes color scheme.
    cityscapes_label_image has the same width and height as image_to_map but only 1-channel (i.e. is grayscale) and contains the labelIds from CityScapes color scheme for each of the pixels.
    """
    cityscapes_color_image = np.zeros(shape=image_to_map.shape).astype(np.uint8) # RGB with cityscapes color scheme
    cityscapes_label_image = np.zeros(shape=image_to_map.shape[:2]).astype(np.uint8) # Grayscale image (1-channel) to store cityscapes labelIds

    for color, label in CARLA_COLOR_CITYSCAPES_LABEL.items():
        cityscapes_color, cityscapes_id = CITYSCAPES_LABEL_TO_COLOR_AND_ID[label]
        indices_to_replace = np.where((image_to_map==list(color)).all(axis=2))
        cityscapes_color_image[indices_to_replace] = cityscapes_color
        cityscapes_label_image[indices_to_replace] = cityscapes_id

    return cityscapes_color_image, cityscapes_label_image

def convert_all_images_to_cityscapes(in_basepath, out_basepath):
    """
    Function that reads all the images the given input path and applies the Carla --> CityScapes color scheme remapping

    Parameters
    ---
    in_basepath: str
        path to the folder where the input images (i.e., images collected via Carla's Semantic Segmentation Camera) are stored

    out_basepath: str
        path to the folder where the output color and grayscale (i.e. labelId) images obtained from the remapping are to be stored

    Returns
    ---
    None
    """
    path_to_images_to_map = glob.glob('{}/*.png'.format(in_basepath))

    for image_path in path_to_images_to_map:
        image_name = os.path.basename(image_path)[:-4]
#         image_name = image_path.split('_')[-1][:-4] # discard the info about the path and the trailing '.png'
        #image_name = image_path.split('/')[-1][:-4]
        print('Mapping: {}.png'.format(image_name))

        image = imageio.imread(image_path)
        image = image[:, :, :3] # convert from Carla's RGBA format to RGB format

        cityscapes_color_image, cityscapes_label_image = convert_single_image_to_cityscapes(image)

        imageio.imwrite('{}/{}gtFine_color.png'.format(out_basepath, image_name), cityscapes_color_image) # output filename format for RGB = aachen_000053_000019_gtFine_color.png
        imageio.imwrite('{}/{}gtFine_labelIds.png'.format(out_basepath, image_name), cityscapes_label_image) # output filename format for LabelID = aachen_000053_000019_gtFine_labelIds.png 

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-folder', type=str, help='path to the folder containing images with Carla\'s SS tags')
    parser.add_argument('--out-folder', type=str, help='path to the folder where the remapped images are to be saved')
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True) # if output folder not found, create it

    convert_all_images_to_cityscapes(args.in_folder, args.out_folder)



















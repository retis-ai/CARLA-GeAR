import glob
import imageio
import numpy as np

#input_basepath = '/data/dataset_rgb'
#output_basepath = '/data/dataset_carla/leftImg8bit/val/Town03DUP'

input_basepath = '/data/rgb'
output_basepath = '/data/dataset_carla/leftImg8bit/billboard/Town03DUP'

img_files = glob.glob('{}/*.png'.format(input_basepath))

for imgpth in img_files:
	filename = imgpth.split('_')[-1][:-4]
	#filename = imgpth.split('/')[-1][:-4]
	print('Processing:', filename)
	img = imageio.imread(imgpth)
	img = img[:, :, :3]
	imageio.imwrite('{}/{}_leftImg8bit.png'.format(output_basepath, filename), img)


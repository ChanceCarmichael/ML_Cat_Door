#let’s create the build_dataset.py script used to build our dataset directory, along with the train and val subdirectories.

# USAGE
# python build_dataset.py
# import necessary packages
import config
from imutils import paths
import logging
import numpy as np
import shutil
import os 

def init_logging():
    '''
    Creates a log found at /debug.log
    '''
    logging.basicConfig(
        filename='debug.log',
        encoding='utf-8',
        level=logging.DEBUG
    )

def copy_images(imagePaths, folder):
	# check if the destination folder exists and if not create it
	if not os.path.exists(folder):
		os.makedirs(folder)
	# loop over the image paths
	for path in imagePaths:
		# grab image name and its label from the path and create
		# a placeholder corresponding to the separate label folder
		imageName = path.split(os.path.sep)[-1]
		label = path.split(os.path.sep)[1]
		labelFolder = os.path.join(folder, label)
		# check to see if the label folder exists and if not create it
		if not os.path.exists(labelFolder):
			os.makedirs(labelFolder)
		# construct the destination image path and copy the current
		# image to it
		destination = os.path.join(labelFolder, imageName)
		print(f"{destination} - {imageName}")
		shutil.copy(path, destination) 

def load_images():
    imagePath = config.DATA_PATH
    return list(paths.list_images(imagePath))

def _main():
    images = load_images()    
    np.random.shuffle(images)
    
    pathLength = len(images)
    valPathsLen = int(pathLength * config.VAL_SPLIT)
    trainPathsLen = pathLength - valPathsLen
    
    trainPaths = images[:trainPathsLen]
    logging.debug(f"{trainPaths}")
    valPaths = images[trainPathsLen:]
    
    copy_images(trainPaths, config.TRAIN)
    copy_images(valPaths, config.VAL)
    
if __name__ == "__main__":
	try:
		init_logging()
		_main()
	except Exception as exc:
		logging.debug(f"exception in main: {exc}")

	'''
	# load all the image paths and randomly shuffle them
	print("[INFO] loading image paths...")
	load_images()
	imagePaths = list(paths.list_images(config.DATA_PATH))
	np.random.shuffle(imagePaths)
	# generate training and validation paths
	valPathsLen = int(len(imagePaths) * config.VAL_SPLIT)
	trainPathsLen = len(imagePaths) - valPathsLen
	trainPaths = imagePaths[:trainPathsLen]
	valPaths = imagePaths[trainPathsLen:]
	# copy the training and validation images to their respective
	# directories
	print("[INFO] copying training and validation images...")
	copy_images(trainPaths, config.TRAIN)
	copy_images(valPaths, config.VAL)
	'''
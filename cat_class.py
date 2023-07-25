import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import imghdr

import file_processing.traverse_files


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode 

#Let’s take a single image name and its annotations from the CSV, 
#in this case row index number 65 for person-7.jpg just as an example. 
#Read it, store the image name in img_name and store its annotations in 
#an (L, 2) array landmarks where L is the number of landmarks in that row.

cat_df = traverse_files('archive')

######Helper Functions########

#Plots feature points on the images. 
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
# plt.figure()
# show_landmarks(io.imread(os.path.join('data/faces/', img_name)),
#                landmarks)
# plt.show()

#Only Runs if running this module. Will not run if imported elsewhere. 
if __name__ == "__main__":


#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


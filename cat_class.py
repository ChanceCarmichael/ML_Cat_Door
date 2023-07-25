import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import imghdr

import file_processing


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode 


#Generate a pandas df of all cat images with annotations. 
cat_df = file_processing.traverse_files('archive')

#Normalize the data to be run through the model. 
n=0
img_name = cat_df.iloc[n, 0]
landmarks = cat_df.iloc[n, 1:]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

######Helper Functions########


#Only Runs if running this module. Will not run if imported elsewhere. 
# if __name__ == "__main__":


#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
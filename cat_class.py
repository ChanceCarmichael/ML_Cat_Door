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

######Helper Functions########

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


#Only Runs if running this module. Will not run if imported elsewhere. 
# if __name__ == "__main__":


#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode 

#ALREADY RUN SEPARATELY TO GENERATE A CSV, NEED TO FIND A WAY TO JUST USE THE DATAFRAME. 
# #Generate a pandas df of all cat images with annotations. 
# cat_df = file_processing.traverse_files('archive')

#Normalize the data to be run through the model. Reshape 
cat_landmarks = pd.read_csv('cat_df.csv')
n=0
img_name = cat_landmarks.iloc[n, 1]
landmarks = cat_landmarks.iloc[n, 3:]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4]))

#Show Landmarks on Image
plt.figure()
show_landmarks(io.imread(os.path.join(img_name)),landmarks)
plt.show()


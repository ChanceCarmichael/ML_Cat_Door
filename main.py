#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
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
import cat_class
from helper_functions import *


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode 

#ALREADY RUN SEPARATELY TO GENERATE A CSV, NEED TO FIND A WAY TO JUST USE THE DATAFRAME. 
# #Generate a pandas df of all cat images with annotations. 
# cat_df = file_processing.traverse_files('archive')

#Normalize the data to be run through the model. Reshape 
cat_landmarks = pd.read_csv('cat_df_new.csv')
n=0
img_name = cat_landmarks.iloc[n, 1]
landmarks = cat_landmarks.iloc[n, 3:]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:9]))

#Show Landmarks on Image
plt.figure()
show_landmarks(io.imread(os.path.join(img_name)),landmarks)
plt.show()

#Instantiate the Class
root_dirs = ['archive/CAT_00', 'archive/CAT_01', 'archive/CAT_02', 
'archive/CAT_03', 'archive/CAT_04', 'archive/CAT_05', 'archive/CAT_06']
catimages = get_catimages(root_dirs)

cat_dataset = cat_class.CatLandmarksDataset(csv_file='cat_df_new.csv', root_dir=catimages)

fig = plt.figure()

for i, sample in enumerate(cat_dataset):
    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_landmarks(**sample)

    if i == 3:
        plt.show()
        break


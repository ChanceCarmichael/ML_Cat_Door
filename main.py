#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import file_processing
from cat_class import *
from helper_functions import *


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
print('First 4 Landmarks: {}'.format(landmarks[:9]))

#Show Landmarks on Image
plt.figure()
show_landmarks(io.imread(os.path.join('cat_images',img_name)),landmarks)
plt.show()

#Instantiate the Class

cat_dataset = CatLandmarksDataset(csv_file='cat_df.csv', root_dir='cat_images')

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

#Resize and compose all cat images so they are uniform. 
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),RandomCrop(224)])

# Apply each of the above transforms on sample.
fig = plt.figure()
sample = cat_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()

#Interate through the Dataset and add the transforms
transformed_dataset = CatLandmarksDataset(csv_file='cat_df.csv',root_dir='cat_images',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

for i, sample in enumerate(transformed_dataset):
    print(i, sample['image'].size(), sample['landmarks'].size())

    if i == 3:
        break

#Need to add in dataloader features. Allows for batching the data, shuffling, and multiprocessing. 
dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=0)

# if you are using Windows, uncomment the next line and indent the for loop.
# you might need to go back and change ``num_workers`` to 0.

# if __name__ == '__main__':
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break

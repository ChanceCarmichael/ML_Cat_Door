import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode 

#Let’s take a single image name and its annotations from the CSV, 
#in this case row index number 65 for person-7.jpg just as an example. 
#Read it, store the image name in img_name and store its annotations in 
#an (L, 2) array landmarks where L is the number of landmarks in that row.

landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

n = 65
img_name = landmarks_frame.iloc[n, 0]
landmarks = landmarks_frame.iloc[n, 1:]
landmarks = np.asarray(landmarks)
landmarks = landmarks.astype('float').reshape(-1, 2)

print('Image name: {}'.format(img_name))
print('Landmarks shape: {}'.format(landmarks.shape))
print('First 4 Landmarks: {}'.format(landmarks[:4])) 

#Convert .cat file to a pandas df. 
def read_cat_file(filename):
    """Reads a .cat file and returns a Pandas DataFrame."""

    df = pd.DataFrame()
    with open(filename, 'r') as f:
        for line in f:
            row = line.split(',')
            df = df.append({
                'item_id': row[0],
                'name': row[1],
                'description': row[2],
                'price': row[3],
                'stock_level': row[4]
            }, ignore_index=True)
    return df

df = read_cat_file('my_cat_file.cat')




#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import imghdr


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode 

#Let’s take a single image name and its annotations from the CSV, 
#in this case row index number 65 for person-7.jpg just as an example. 
#Read it, store the image name in img_name and store its annotations in 
#an (L, 2) array landmarks where L is the number of landmarks in that row.


######Helper Functions########

#Convert .cat file to a pandas df. 
def read_cat_file(filename):
    """Reads a .cat file and returns a Pandas DataFrame."""

    df = pd.DataFrame()
    with open(filename, 'r') as f:
        for line in f:
            row = line.split(' ')
            df = df.append({
                'Left Eye': row[1:3],
                'Right Eye': row[3:5],
                'Mouth': row[5:7],
                'Left Ear-1': row[7:9],
                'Left Ear-2': row[9:11],
                'Left Ear-3': row[11:13],
                'Right Ear-1': row[13:15],
                'Right Ear-2': row[15:17],
                'Right Ear-3': row[17:19]
            }, ignore_index=True)
    return df
# annotations = read_cat_file('archive/CAT_00/00000001_000.jpg.cat')
# print(annotations['Left Eye'][0][1])

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

#File Traversal
def traverse_files(path):
    for root, directories, files in os.walk(path):
        for file in files:
            print(root,'\n',file,'\n-----------')

#Only Runs if running this module. Will not run if imported elsewhere. 
if __name__ == "__main__":
	traverse_files('archive')


#https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


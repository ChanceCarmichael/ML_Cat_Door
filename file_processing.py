import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import imghdr

#Create the Dictionary for the annotations 
annotations = {
"Filename":[], 
"Points":[], 
"LeftEye":[], 
"RightEye":[], 
"Mouth":[], 
"LeftEar1":[], 
"LeftEar2":[],
"LeftEar3":[],
"RightEar1":[], 
"RightEar2":[], 
"RightEar3":[] 
}

#Create the dataframe 
df = pd.DataFrame(annotations)

print(df)

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

#File Traversal
def traverse_files(path):
    for root, directories, files in os.walk(path):
        for file in files:
            print(root,'\n',file,'\n-----------')

#Only Runs if running this module. Will not run if imported elsewhere. 
if __name__ == "__main__":
	traverse_files('archive')
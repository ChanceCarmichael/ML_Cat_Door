import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import imghdr

# #Create Empty Annotations Lists
# filename_pts = [] 
# points_pts = [] 
# lefteye_pts = []
# righteye_pts = [] 
# mouth_pts = []
# leftear1_pts = []
# leftear2_pts = [] 
# leftear3_pts = []
# rightear1_pts = [] 
# rightear2_pts = [] 
# rightear3_pts = []

# #Create the Dictionary for the annotations 
# annotations = {
# "Filename":filename_pts, 
# "Points":points_pts, 
# "LeftEye":lefteye_pts, 
# "RightEye":righteye_pts, 
# "Mouth":mouth_pts, 
# "LeftEar1":leftear1_pts, 
# "LeftEar2":leftear2_pts,
# "LeftEar3":leftear3_pts,
# "RightEar1":rightear1_pts, 
# "RightEar2":rightear2_pts, 
# "RightEar3":rightear3_pts 
# }

# df = pd.DataFrame(annotations)

#Convert .cat file to a pandas df. 

#File Traversal
def traverse_files(path):
    df = pd.DataFrame()

    for root, directories, files in os.walk(path):
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if file_extension == ".cat":
                #Read and sort cat data into df
                with open(root+'/'+file, 'r') as f:
                    for line in f:
                        file_name = os.path.splitext(file)[0]
                        row = line.split(' ')
                        df = df.append({
                            'Filename': file_name,
                            'Points': row[0],
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
            else:
                continue
    return df

#Only Runs if running this module. Will not run if imported elsewhere. 
if __name__ == "__main__":
	print(traverse_files('archive'))
    

    
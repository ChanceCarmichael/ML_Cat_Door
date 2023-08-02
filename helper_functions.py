import matplotlib.pyplot as plt
from cat_class import CatLandmarksDataset
import os 
import shutil

#--------------Helper Functions---------------#

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(1)  # pause a bit so that plots are updated

def get_catimages(root_dirs):
    datasets = []
    for root_dir in root_dirs:
        dataset = CatLandmarksDataset(csv_file='cat_df_new.csv', root_dir=root_dir)
        datasets.append(dataset)
    return datasets

def separate_jpgs(source_dir, destination_dir):
    for root, directories, files in os.walk(source_dir):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == ".jpg":
                shutil.copy(os.path.join(root, file), destination_dir) 
    print("Complete")


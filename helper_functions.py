import matplotlib.pyplot as plt
from cat_class import CatLandmarksDataset
import os 
import shutil
from torchvision import utils

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

# Helper function to show a batch
def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')


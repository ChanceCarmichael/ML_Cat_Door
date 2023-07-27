import matplotlib.pyplot as plt

#--------------Helper Functions---------------#

def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(5)  # pause a bit so that plots are updated

def get_catimages(root_dirs):
    datasets = []
    for root_dir in root_dirs:
        dataset = CatLandmarksDataset(csv_file='cat_df_new.csv', root_dir=root_dir)
        datasets.append(dataset)
    return datasets

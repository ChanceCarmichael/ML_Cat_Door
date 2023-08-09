#PyTorch allows us to easily construct DataLoader objects from images stored in directories on disk. 

# import the necessary packages
from . import config
from torch.utils.data import DataLoader
from torchvision import datasets
import os 

def get_dataloader(rootDir, transforms, batchSize, shuffle=True):
	# create a dataset and use it to create a data loader
	ds = datasets.ImageFolder(root=rootDir,transform=transforms)
	loader = DataLoader(ds, batch_size=batchSize,
		shuffle=shuffle,
		num_workers=os.cpu_count(),
		pin_memory=True if config.DEVICE == "cuda" else False)
	# return a tuple of  the dataset and the data loader
	return (ds, loader)
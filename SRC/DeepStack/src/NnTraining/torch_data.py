'''
	Helper functions for reading and parsing data from TFRecords
'''
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import random


class CustomDataset(Dataset):
	def __init__(self, filenames, x_shape, y_shape):
		self.filenames = filenames
		self.x_shape = x_shape
		self.y_shape = y_shape

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		# Load file (assuming it's a binary file with serialized tensors)
		filename = self.filenames[idx]
		with open(filename, 'rb') as f:
			serialized = f.read()

		# Parsing logic: assuming serialized is in a compatible format
		x, y = self.parse_fn(serialized)
		return x, y

	def parse_fn(self, serialized):
		'''
        Replace this function with the appropriate deserialization for your data.
        '''
		# Example deserialization (replace with your actual format)
		features = np.frombuffer(serialized, dtype=np.float32)
		x = features[:np.prod(self.x_shape)].reshape(self.x_shape)
		y = features[np.prod(self.x_shape):].reshape(self.y_shape)
		return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def create_iterator(filenames, train, x_shape, y_shape, batch_size, num_workers=os.cpu_count()):
	'''
    Creates a PyTorch DataLoader for training or testing.

    @param: [str,...] :Filenames for the data files.
    @param: bool      :Boolean whether training (True) or testing (False).
    @param: [1]       :input shape (not including batch size), ex: [224,224,3] if img
    @param: [1]       :output shape (not including batch size), ex: [224,224,3] if img
    @param: int       :return batches of this size.
    @param: int       :number of workers to use for data loading.
    @return: PyTorch DataLoader.
    '''
	dataset = CustomDataset(filenames, x_shape, y_shape)

	# If training, enable shuffling
	shuffle = train

	# Create PyTorch DataLoader
	data_loader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=True,
		drop_last=True
	)

	return data_loader

#

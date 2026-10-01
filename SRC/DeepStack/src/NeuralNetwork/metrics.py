'''
	Computes a Huber loss for neural net training and evaluation.
	Computes the loss across buckets.
'''

import torch
import numpy as np

# main loss
def BasicHuberLoss(delta=1.0):
	def loss(y_true, y_pred):
		return torch.nn.SmoothL1Loss()(y_true, y_pred)
	return loss


# used only as metric
def masked_huber_loss(y_true, y_pred):
	loss = torch.nn.SmoothL1Loss()(y_true, y_pred)
	zero = torch.nn.init.constant(0.0, dtype=np.float32)
	batch_size = np.cast( np.shape(y_true)[0], dtype=np.float32 ) # 1024
	feature_size = np.cast( np.shape(y_true)[1], dtype=np.float32 ) # 1326*2
	mask = np.where( np.equal(y_true, zero), np.zeros_like(y_true), np.ones_like(y_true) )
	loss_multiplier = (batch_size * feature_size) / np.sum(mask)
	return loss * loss_multiplier




#

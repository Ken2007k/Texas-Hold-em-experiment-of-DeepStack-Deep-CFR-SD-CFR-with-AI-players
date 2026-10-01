'''
	Wraps the calls to the final neural net.
'''
import os
import torch
import torch.nn as nn
import numpy as np
from Settings.arguments import arguments
from Settings.constants import constants
from Game.card_to_string_conversion import card_to_string
from NeuralNetwork.metrics import BasicHuberLoss, masked_huber_loss

class ValueNn():
	def __init__(self, street, pretrained_weights=False, approximate='root_nodes', verbose=1):
		'''
		@param: int  :current street/round
		@param: bool :to load pretrained model or init random weights
		@param: str  :approximate current street "root_nodes"/"leaf_nodes"
		@param: int  :display output if >0
		'''
		# set directories
		self.approximate = approximate # set to approximate leaf or root nodes of specified street
		street_name = card_to_string.street_to_name(street)
		self.model_dir_path = os.path.join(arguments.model_path, street_name)
		model_name = '{}.{}.hdf5'.format(arguments.model_filename, self.approximate)
		self.model_path = os.path.join(self.model_dir_path, model_name)
		# set input, output shapes
		self._set_shapes()
		# load model or create one
		if pretrained_weights:
			self.torch_model = torch.load( self.model_path,
								   map_location = {'loss':BasicHuberLoss(delta=1.0),
													 'masked_huber_loss':masked_huber_loss} )
		else: # create torch model
			self.torch_model = self._build_net()
		# print architecture summary
		if verbose > 0:
			print('NN architecture:',self.torch_model)
			#self.keras_model.summary()


	def predict(self, inputs, out):
		''' Gives the neural net output for a batch of inputs
		@param: [b,nnI] :tensor containing b batches instances of neural net inputs
		@param: [b,nnO] :tensor in which to store b batches of neural net outputs
		'''
		#with self.graph.as_default():
		with torch.no_grad():
			total_elements, batch_size = inputs.shape[0], 10000
			for i in range(0, total_elements, batch_size):
				start, end = i, i + batch_size
				end = end if end < total_elements else total_elements
				out[ start:end, : ] = self.torch_model(inputs[ start:end, : ])


	def _set_shapes(self):
		''' sets self.x_shape and self.y_shape '''
		num_ranks, num_suits, num_cards = constants.rank_count, constants.suit_count, constants.card_count
		num_hands, num_players = constants.hand_count, constants.players_count
		# input and output parameters
		num_output = num_hands * num_players
		num_input = num_output + 1 + num_cards + num_suits + num_ranks
		self.x_shape = [num_input]
		self.y_shape = [num_output]


	def _build_net(self):
		''' creates neural network architecture '''
		num_output = self.y_shape[0]


		class CustomModel(nn.Module):
			def __init__(self):
				super(CustomModel, self).__init__()
				# Feed-forward part: Define layers based on arguments.num_neurons
				self.feed_forward = nn.ModuleList()
				input_size = self.x_shape[1]

				for i, num_neurons in enumerate(arguments.num_neurons):
					dense = nn.Linear(input_size, num_neurons)
					dropout = nn.Dropout(0.10)
					batch_norm = nn.BatchNorm1d(num_neurons)
					prelu = nn.PReLU()
					input_size = num_neurons  # Update input size for next layer
					self.feed_forward.extend([dense, dropout, batch_norm, prelu])

				# Output layer
				self.output_layer = nn.Linear(arguments.num_neurons[-1], num_output)

			def forward(self, x):
				# Slicing off pot size and board ([1, hands x 2 + pot_size + board] -> [1, hands x 2])
				ranges = x[:, :num_output]

				# Reconstruct mask for not possible ranges (where ranges are 0)
				mask = np.where(ranges > 0, np.ones_like(ranges), np.zeros_like(ranges))

				# Feed-forward part
				ff = x
				for layer in self.feed_forward:
					ff = layer(ff)

				# Final feed-forward output
				ff = self.output_layer(ff)

				# Mask neural network output
				values = ff * mask

				# Compute zero-sum output
				estimated_value = np.matmul(values, ranges.T).diagonal() / 2
				estimated_value = estimated_value.unsqueeze(1)  # Align dimensions for broadcasting

				m_output = values - estimated_value

				return m_output

		# Create model instance
		model = CustomModel()
		return model





#

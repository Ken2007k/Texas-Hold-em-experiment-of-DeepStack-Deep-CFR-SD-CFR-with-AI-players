import os
import torch
import random
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Settings.arguments import arguments
from NnTraining.tf_data import create_iterator
from NeuralNetwork.value_nn import ValueNn
from NeuralNetwork.metrics import BasicHuberLoss, masked_huber_loss
class Train(ValueNn):
    def __init__(self, data_dir_list, street):
        '''
        @param: [str,...] :list of paths to directories that contains data files
        '''
        # Set up model from ValueNn
        super().__init__(street)
        # Resume model if it exists
        if os.path.exists(self.model_path):
            print('LOADING PREVIOUS MODEL...')
            self.torch_model = torch.load(self.model_path)
        else:  # Compile model
            print('COMPILING MODEL...')
            self.torch_model = self.compile_pytorch_model()

        # Set up read paths for train/valid datasets
        self.records = [f.path for dirpath in data_dir_list for f in os.scandir(dirpath)]

    def compile_pytorch_model(self):
        ''' Compiles PyTorch model '''
        loss = BasicHuberLoss(delta=1.0)
        optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=arguments.learning_rate, betas=(0.9, 0.999), weight_decay=0.0)
        return self.torch_model, optimizer

    def train(self, num_epochs, batch_size, verbose=1, validation_size=0.1, start_epoch=0):
        ''' Trains model with specified parameters '''
        # Get list of train and validation set filenames
        num_valid_files = int(len(self.records) * validation_size) if validation_size < 1 else validation_size
        train_filenames = self.records[:-num_valid_files]
        random.shuffle(train_filenames)
        valid_filenames = self.records[-num_valid_files:]

        # Create PyTorch data loaders (assuming create_iterator loads the data)
        train_loader = create_iterator(filenames=train_filenames, train=True,
                                       batch_size=batch_size, x_shape=self.x_shape, y_shape=self.y_shape)
        valid_loader = create_iterator(filenames=valid_filenames, train=False,
                                       batch_size=batch_size, x_shape=self.x_shape, y_shape=self.y_shape)

        # Count number of elements in both sets
        num_train_elements = len(train_filenames) * arguments.batch_size
        num_valid_elements = len(valid_filenames) * arguments.batch_size

        # Training model
        print('Training model...')
        for epoch in range(start_epoch, num_epochs):
            self._train_epoch(train_loader, epoch, num_train_elements, batch_size, verbose)
            self._validate_epoch(valid_loader, epoch, num_valid_elements, batch_size)

    def _train_epoch(self, train_loader, epoch, num_train_elements, batch_size, verbose):
        ''' Helper function to train the model for one epoch '''
        self.torch_model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            self.optimizer.zero_grad()
            output = self.torch_model(data)
            loss = masked_huber_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if verbose:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{num_train_elements} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    def _validate_epoch(self, valid_loader, epoch, num_valid_elements, batch_size):
        ''' Helper function to validate the model after each epoch '''
        self.torch_model.eval()
        valid_loss = 0
        with torch.no_grad():
            for data, target in valid_loader:
                output = self.torch_model(data)
                valid_loss += masked_huber_loss(output, target).item()

        valid_loss /= num_valid_elements
        print(f'\nValidation set: Average loss: {valid_loss:.6f}\n')

    def update_belief(self, belief_vector):
        '''
        Updates the belief vector in CFR (Counterfactual Regret Minimization).
        @param: belief_vector : the belief vector representing strategy probabilities
        '''
        # Use the trained model to update the belief vector
        with torch.no_grad():
            belief_tensor = torch.Tensor(belief_vector)
            new_beliefs = self.torch_model(belief_tensor)

        updated_beliefs = new_beliefs.numpy()  # Assuming the belief is represented in a numpy array
        print(f"Updated Belief Vector: {updated_beliefs}")
        return updated_beliefs

    def evaluate_strategy(self, state):
        '''
        Evaluates the current strategy based on the given state.
        @param: state : the current state of the game/environment
        @return: strategy evaluation as a probability distribution
        '''
        with torch.no_grad():
            state_tensor = torch.Tensor(state)
            strategy = self.torch_model(state_tensor)

        strategy_eval = strategy.numpy()
        print(f"Strategy Evaluation: {strategy_eval}")
        return strategy_eval

    def produce_strategy(self, state):
        '''
        Produces a strategy (policy) based on the current model's estimation for the given state.
        @param: state : the current state of the game/environment
        @return: a strategy (policy) derived from the model
        '''
        with torch.no_grad():
            state_tensor = torch.Tensor(state)
            strategy = self.torch_model(state_tensor)

        strategy_output = strategy.softmax(dim=-1)  # Use softmax to normalize to probabilities
        strategy_output = strategy_output.numpy()
        print(f"Produced Strategy: {strategy_output}")
        return strategy_output


class PyTorchTensorBoard:
    ''' PyTorch TensorBoard logger for training/validation losses '''
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_epoch_metrics(self, epoch, train_loss, valid_loss):
        ''' Log metrics for TensorBoard visualization '''
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/validation', valid_loss, epoch)

    def close(self):
        ''' Close the writer after training '''
        self.writer.close()


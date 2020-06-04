
# This script use deep learning to train an image classifier from a training set of flower images with labeled 
# classes of flowers. There are 102 different classes in total.
# The model will be trained through transfer learning based on a pre-trained neural network model of the user's choice. 
# It will print out training loss, validation loss, and validation accuracy as the network trains.

# The basic usage is to run the command "python train.py data_directory"in the terminal. 
# A few options are available: 
# set directory to save the checkpoint using --save_dir save_directory;
# choose architecture from vgg13 and alexnet using --arch;
# set hyperparameters using --learning_rate, --hidden_units, and --epochs;
# choose GPU for training by --gpu.
# An example command is in the bash file 'model_training_example.sh' and is also copied here:
# python train.py ./flowers --save_dir ./ --arch vgg13 --learning_rate 0.001 --hidden_units 1024 --epochs 5 --gpu
# Or you can just run "sh model_training_example.sh" in the terminal.


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from PIL import Image
from utility import *
import argparse



def main():

	# take arguments from the command
	in_arg = get_input_args()
    
	if in_arg.data_dir is None:
		print("Please re-type your command because the data directory is missing!\n")
		return

	# set up data
	train_data, valid_data, trainloader, validloader = data_setup(in_arg.data_dir)

	# set up model structure
	model, criterion, optimizer = model_setup(in_arg.arch, in_arg.hidden_units, in_arg.learning_rate)
	
	if model == None:
		print('Please re-type your command because the architecture has to be either vgg13 or alexnet!\n')
		return

	# set up device: GPU or CPU
	device = torch.device("cuda" if (torch.cuda.is_available() and in_arg.gpu == True) else "cpu")
	print('Using {} for training...'.format(device))

	model.to(device)

	# train the model
	model = model_training(model, criterion, optimizer, trainloader, validloader, device, in_arg.epochs)

	# save the checkpoint
	save_checkpoint(in_arg.arch, model, in_arg.save_dir, train_data)

	print("Model trained! Checkpoint saved!\n")


# Call to main function to run the program
if __name__ == "__main__":
	main()
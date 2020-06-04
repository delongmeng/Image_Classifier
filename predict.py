
# This script will take any flower image of the user's choice, and used the trained deep learning model
# to predict the class of the flower image. There are a total of 102 different kinds of flowers.
# This program will print out the k (defined by the user) most likely categories with corresponding possibilities.

# The basic usage is to run the command "python predict.py image_path checkpoint" in the terminal.
# A few options are available:
# define the k number using --top_k;
# add a mapping of categories to real names using --category_names;
# use GPU for inference by --gpu.
# An example command is in the bash file "model_predicting_example.sh" and is also copied here:
# python predict.py uploaded_images/french_marigold.jpg checkpoint.pth --category_name cat_to_name.json --top_k 5 --gpu
# Alternatively, you can just run "sh model_predicting_example.sh" in the terminal.


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



def main():

	# take arguments from the command
	in_arg = get_input_args_predict()

	# load the mapping between flower name and class
	with open(in_arg.category_names, 'r') as f:
	    cat_to_name = json.load(f)

	# set up device: GPU or CPU
	device = torch.device("cuda" if (torch.cuda.is_available() and in_arg.gpu == True) else "cpu")
	print('Using {} for predicting...'.format(device))

	# set up model structure
	model = load_checkpoint(in_arg.checkpoint)
	model.to(device)

	# make prediction
	predict(in_arg.image_path, model, device, in_arg.top_k, cat_to_name)


# Call to main function to run the program
if __name__ == "__main__":
    main()
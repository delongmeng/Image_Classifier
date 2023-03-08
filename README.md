

# Image Classifier

## Flower image classification using PyTorch neural network models

### Overview

Deep learning is one of the most powful tools for image recoganization. This project aims to showcase how neural network deep learning models can be trained in a PyTorch platform and used to make prediction. The application here is to recognize 102 different species of flowers. The original dataset can be found here: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html. All the images are in the `flowers` folder. In the training set there are more than 6000 images of flowers. Considering the relatively small sample size, it will be a good idea to perform a transfer learning, in which a pretrained neural network model (such as VGG and Alexnet) will be borrowed, and only the last several layers will be trained on the flower training set to fine-tune the weights. ReLU activation and Softmax output are used. Dropout method is used to avoid over-fitting. 

During the training process, the model was also tested on the validation dataset, and the loss and accuracy were displayed. After the model was trained, it was applied on the test dataset and achieved an accuracy rate of around 80%. Images can also be provided by user for the prediction of flower species. The top k (for example top 5) classes will be returned with corresponding probabilities.

GPU is recommended to use for the training process, and the estimated training time using GPU is around 30-60 minutes. Once training is done, the model will be saved. Using the saved model, prediction can be made very fast.


### Notebook file

Please refer to the Jupyter Notebook file `Image_Classifier.ipynb` for the implementation of the model training and prediction. The VGG19 model was used as the pre-trained model. The trained model was saved in `DM_vgg19_checkpoint.pth`, which can be loaded to make quick prediction on any input images. The input image and prediction result are nicely displayed in a plot. For example, for a flower image randomly downloaded from internet, this model successfully predicted its category. Please refer to the end of the notebook, as well as the screenshot `predicting_uploaded_images_output.jpg`.


### Command line scripts

For command line applications, there are three scripts: `train.py`, `predict.py`, and `utility.py`. Running `train.py` will train the model using the training dataset; on the other hand, the `predict.py` will load the trained model and make prediction for input image. `utility.py` is a helper module containing all useful functions for training and prediction.

You can directly run command such as `python train.py` or `python predict.py` in the terminal. Some arguments need to be specified and other ones are optional. For example, two alternative pre-trained models `Vgg13` and `Alexnet` can be chosen from. `checkpoint.pth` was the saved structure of a trained model based on `Vgg13`, and can be used for prediction part without need to do the training. For details, see below:


### train.py
This script use deep learning to train an image classifier from a training set of flower images with labeled classes of flowers. There are 102 different classes in total. The model will be trained through transfer learning based on a pre-trained neural network model of the user's choice. It will print out training loss, validation loss, and validation accuracy as the network trains.

The basic usage is to run the command `python train.py data_directory` in the terminal. A few options are available:  

- set directory to save the checkpoint using `--save_dir save_directory`;  
- choose architecture from vgg13 and alexnet using `--arch`;  
- set hyperparameters using `--learning_rate`, `--hidden_units`, and `--epochs`;  
- choose GPU for training by `--gpu`.  

### Example for running train.py
An example command is in the bash file `model_training_example.sh` and is also copied here:

```
python train.py ./flowers --save_dir ./ --arch vgg13 --learning_rate 0.001 --hidden_units 1024 --epochs 5 --gpu
```

Or you can just run `sh model_training_example.sh` in the terminal.


### predict.py
This script will take any flower image of the user's choice, and used the trained deep learning model to predict the class of the flower image. There are a total of 102 different kinds of flowers. This program will print out the k (defined by the user) most likely categories with corresponding possibilities.

The basic usage is to run the command `python predict.py image_path checkpoint` in the terminal. A few options are available:  

- define the k number using `--top_k`;
- add a mapping of categories to real names using `--category_names`;
- use GPU for inference by `--gpu`.

### Example for running predict.py
An example command is in the bash file `model_predicting_example.sh` and is also copied here:

```
python predict.py uploaded_images/french_marigold.jpg checkpoint.pth --category_name cat_to_name.json --top_k 5 --gpu
```

Alternatively, you can just run `sh model_predicting_example.sh` in the terminal.



### Author: 
Delong Meng, delongmeng@hotmail.com

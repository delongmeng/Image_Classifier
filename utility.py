
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
import argparse




def get_input_args():
    '''
    This function parses the arguments for train.py. 
    '''    

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('data_dir', type = str, 
                    help = 'Data directory is a mandatory argument')
    parser.add_argument('--save_dir', type = str, default = './', 
                    help = 'directory to save checkpoint of the trained model')
    parser.add_argument('--arch', type = str, default = 'vgg13', 
                    help = 'the CNN model architecture to use (default: vgg13)')
    parser.add_argument('--learning_rate', type = float, default = 0.01, 
                    help = 'the learning rate for model training')    
    parser.add_argument('--hidden_units', type = int, default = 512, 
                    help = 'the number of hidden units for model training')        
    parser.add_argument('--epochs', type = int, default = 20, 
                    help = 'the number of epochs for model training')  
    parser.add_argument('--gpu', action = 'store_true', 
                    help = 'whether or not use GPU for training')  

    return parser.parse_args()   




def get_input_args_predict():
    '''
    This function parses the arguments for predict.py. 
    ''' 

    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('image_path', type = str, 
                    help = 'Image path is a mandatory argument')
    parser.add_argument('checkpoint', type = str, default = 'checkpoint.pth',
                    help = 'Checkpoint is a mandatory argument')   
    parser.add_argument('--top_k', type = int, default = 3, 
                    help = 'The top K most likely classes will be provided.')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', 
                    help = 'The json file of the mapping between categories and corresponding names.')
    parser.add_argument('--gpu', action = 'store_true', 
                    help = 'whether or not use GPU for training')  

    return parser.parse_args()   




def data_setup(data_dir):
    '''
    Arg: directory of data
    Return: datasets and data loaders
    '''
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    return train_data, valid_data, trainloader, validloader




def model_setup(arch, hidden_units, learning_rate):
    '''
    Arg: architecture of the model (vgg13 or alexnet), and the hyperparameters for the network 
    (hidden units and learning rate)
    Return: the model modified for the training purpose
    '''
    if arch == 'vgg13':        
        model = models.vgg13(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(0.5)),
                              ('fc2', nn.Linear(4096, hidden_units)),
                              ('relu2', nn.ReLU()),
                              ('dropout2', nn.Dropout(0.5)),
                              ('fc3', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

        model.classifier = classifier

    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(9216, 4096)),
                              ('relu1', nn.ReLU()),
                              ('dropout1', nn.Dropout(0.5)),
                              ('fc2', nn.Linear(4096, hidden_units)),
                              ('relu2', nn.ReLU()),
                              ('dropout2', nn.Dropout(0.5)),
                              ('fc3', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

        model.classifier = classifier

    else:
        return None


    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    return model, criterion, optimizer




def model_training(model, criterion, optimizer, trainloader, validloader, device, epochs = 10, print_every = 20):
    '''
    This function trains the model over a certain number of epochs and print out training loss, validation loss
    and accuracy every certain steps.
    Arg: model setups, train and validation data, device (GPU or CPU), epochs, and number of steps for the printout.
    Return: None
    '''

    steps = 0

    # train_losses, valid_losses, accuracies = [], [], []

    for epoch in range(epochs):

        train_loss = 0

        for inputs, labels in trainloader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if steps % print_every == 0:
                valid_loss, accuracy = 0, 0

                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:          

                        inputs, labels = inputs.to(device), labels.to(device)
                        log_ps = model.forward(inputs)
                        loss = criterion(log_ps, labels)
                        valid_loss += loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                train_loss = train_loss/print_every   
                valid_loss = valid_loss/len(validloader)                
                accuracy = accuracy/len(validloader)
                
                print('Epoch: {}/{}..'.format(epoch+1, epochs),
                     'Training Loss: {:.3f}..'.format(train_loss),
                     'Validation Loss: {:.3f}..'.format(valid_loss),
                     'Validation Accuracy: {:.1f}%'.format(accuracy*100))

                train_loss = 0
                model.train()

    return model




def save_checkpoint(arch, model, save_dir, train_data):
    '''
    This function saves the trained model as a checkpoint file.
    Arguments: Parameters of the model and saving path.
    Returns: None.
    '''

    path = save_dir + 'checkpoint.pth'
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'architecture': arch,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, path)




def load_checkpoint(filepath = 'checkpoint.pth'):
    '''
    This function loads the saved checkpoint file.
    Arguments: the path of the checkpoint file.
    Returns: None.
    '''

    checkpoint = torch.load(filepath)
    
    if checkpoint['architecture'] == 'vgg13':        
        model = models.vgg13(pretrained=True)
    elif checkpoint['architecture'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model


    
    
def process_image(image_path):
    '''
    This function open an image as a PIL image, scales, crops, and normalizes it for a PyTorch model.
    Arguments: the path of the image file.
    Returns: processed image in a PyTorch model.
    '''
    
    image = Image.open(image_path)
    image_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                      [0.229, 0.224, 0.225])])
    
    transformed_image = image_transforms(image)

    return transformed_image   
    

    
    


def predict(image_path, model, device, topk, cat_to_name):
    ''' 
    This function predicts the class (or top k classes) of an image using a trained deep learning model.
    Arg: path of image, model, device (GPU or CPU), topk, cat_to_name mapping.
    Returns: None.
    '''
    processed_image = process_image(image_path)
    processed_image = processed_image.unsqueeze_(0)
    processed_image = processed_image.float()
    
    model.eval()
        
    if device == torch.device('cuda'):
        with torch.no_grad():
            output = model.forward(processed_image.cuda())
        ps = torch.exp(output)
        probs, indices = ps.topk(topk)
        probs = probs.cpu()
        indices = indices.cpu()
    else:
        with torch.no_grad():
            output = model.forward(processed_image)    
        ps = torch.exp(output)
        probs, indices = ps.topk(topk)

    
    probs = probs.numpy()[0]
    indices = indices.numpy()[0]    
    
    mapping = {val: key for key, val in model.class_to_idx.items()}
    classes = [mapping[index] for index in indices]
    
    class_names = [cat_to_name[str(key)] for key in classes]
    
    print('\nPrediction results:')
    for i in range(len(probs)):
        print('This image is predicted to be {} (class {}) with a probility of {}%.'.format(class_names[i], classes[i], round(probs[i]*100, 2)))

    

    






    
    
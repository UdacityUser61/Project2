import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from t_10_t import primaryloader_model
import json
import argparse


#Defining command line arguments:
parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='Path to image to predict')
parser.add_argument('--checkpoint', type=str, help='File containing my trained module to use for predicting')
parser.add_argument('--topk', type=int, help='Returning topk predictions')
parser.add_argument('--labels', type=str, help='JSON file for mapping labels with flower names')
args, _ = parser.parse_known_args()
filepath='atemp1.pth'
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = fc_model.Network(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    model.load_state_dict(checkpoint['state_dict'])   
    return model
model=load_checkpoint('atemp1.pth')

def process_image(image):
        # TODO: Process a PIL image for use in a PyTorch model
 
    dataset = torchvision.datasets.DatasetFolder()
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
  
    return dataset
	
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    trans1 = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])  
    testset = image_path
    dataset = torchvision.datasets.DatasetFolder(image_path,transform=trans1 )
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    data_iter = iter(testloader)
    image, label = next(iter(trainloader))
    img = images[1]
    img=process_image(img)

    # TODO: Calculate the class probabilities (softmax) for img
    ps = torch.exp(model(img))
    top_p, top_class = torch.topk(ps, topk, dim=1)
    return top_p, top_class, img	
	
	
# TODO: Display an image along with the top 5 classes
image_path='flowers/test'

probs, classes, img = predict(image_path, model)
print(probs)
print(classes)



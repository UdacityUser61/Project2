import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from collections import OrderedDict
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

def arg_parser():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=1)
    parser.add_argument('--gpu', dest="gpu", action="store", default="gpu")
    args = parser.parse_args()
    return args

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

def train_transform(train_dir):
    data_transforms_train =  transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(data_dir + '/train', transform=data_transforms_train)
    return train_data

def valid_transform(valid_dir):
    data_transforms_valid = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=data_transforms_valid)
    return valid_data

def test_transform(test_dir):
    data_transforms_test = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(data_dir + '/test', transform=data_transforms_test)
    return test_data

#transform images
data_transforms_train =  transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
data_transforms_valid = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

data_transforms_test = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=data_transforms_train)
valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms_valid)
test_data = datasets.ImageFolder(test_dir, transform=data_transforms_test)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(test_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)	


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def primaryloader_model(architecture="vgg16"):
    model = models.vgg16(pretrained=True)
    model.name = "vgg16"
    for param in model.parameters():
        param.requires_grad = False 
    return model

def loader_model(architecture="vgg16"):
    model = models.vgg16(pretrained=True)
    return model

def initial_classifier(model, hidden_units):
    classifier = nn.Sequential(OrderedDict([
                ('inputs', nn.Linear(25088, 4096)),
                ('relu1', nn.ReLU()),
                ('dropout',nn.Dropout(0.2)), 
                ('hidden_layer1', nn.Linear(4096, 1024)), 
                ('relu2',nn.ReLU()),
                ('hidden_layer2', nn.Linear(1024,512)), 
                ('relu2',nn.ReLU()),
                ('hidden_layer3',nn.Linear(512,102)), 
                ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    return classifier





def network_trainer(model, trainloader, testloader, device, 
                  criterion, optimizer, epochs, print_every, steps):
    
    if type(epochs) == type(None):
        epochs = 3
        print("Number of Epochs specificed as 12.")    
 
    print("Training process initializing ...\n")

    # Train Model
    for epoch in range(epochs):
        running_loss = 0
        model.train() 
        
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)
            
                print("Epoch: {}/{} | ".format(epoch+1, epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(testloader)),
                     "Validation Accuracy: {:.4f}".format(accuracy/len(testloader)))
            
                running_loss = 0
                model.train()

    return model

def validation(model, valid_loader, criterion):
    model.to (device)

    valid_loss = 0
    accuracy_v = 0
    for inputs, labels in valid_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        logps = model(inputs)
        batch_loss = criterion(logps, labels)
        valid_loss += batch_loss.item()
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1,dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy_v += torch.mean(equals.type(torch.FloatTensor)).item()
            
    return valid_loss, accuracy_v

def validate_model(model, testloader, device):
   # Do validation on the test set
    correct,total = 0,0
    with torch.no_grad():
        model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test images is: %d%%' % (100 * correct / total))

    
# Function initial_checkpoint(model, save_Dir, train_data) saves the model at a defined checkpoint
def initial_checkpoint(model, save_dir, train_data, optimizer):
    # Save model at checkpoint
    if type(save_dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
        if isdir(save_dir):
            model.class_to_idx = image_datasets['train'].class_to_idx

    model.class_to_idx = train_data.class_to_idx 
    model.name='vgg16'
    model.class_to_idx = train_data.class_to_idx
    checkpoint= {'classifier': model.classifier,
                 'optimizer_state_dict':optimizer.state_dict(),
                 'state_dict':model.state_dict(),
                 'mapping':    model.class_to_idx
                }

#'model_type': model.name,

    torch.save(checkpoint, 'atemp1.pth')

def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    model = primaryloader_model(architecture=args.arch)
    
    model.classifier = initial_classifier(model, hidden_units=args.hidden_units)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device);
    
    
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    print_every = 30
    steps = 0
    
    trained_model = network_trainer(model, trainloader, validloader, device, criterion, optimizer, args.epochs, print_every, steps)
    
    print("\nTraining process is completed!!")
    
    validate_model(trained_model, testloader, device)
   
    initial_checkpoint(trained_model, args.save_dir, train_data,optimizer)
if __name__ == '__main__': main()



#model = loader_model(architecture=args.arch)
#model.classifier = initial_classifier(model, hidden_units=args.hidden_units)
    

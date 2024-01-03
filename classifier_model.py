import torch 
from torch import nn
from torchvision import models
import torch.optim.lr_scheduler as lr_scheduler


def build_model(model_name='resnet50', hidden_layer = 1000):
    if model_name == 'resnet50':
    
        model = models.resnet50(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
    
        in_features = model.fc.in_features

        classifier = nn.Sequential(
                                  nn.BatchNorm1d(num_features=in_features),
                                  nn.Linear(in_features, hidden_layer),
                                  nn.ReLU(), 
                                  nn.BatchNorm1d(num_features=hidden_layer), 
                                  nn.Linear(hidden_layer,102),
                                  nn.LogSoftmax(dim = 1)  
                                )  
                                    
        model.fc = classifier
        
    elif model_name =='densenet_121':
        
        model = models.densenet121(pretrained=True)
        
        for param in model.parameters():
            param.requires_grad = False

        in_features = model.classifier.in_features

        classifier= nn.Sequential(
                                  nn.BatchNorm1d(num_features=in_features),
                                  nn.Linear(in_features, hidden_layer),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(num_features=hidden_layer),
                                  nn.Dropout(0.2),  
                                  nn.Linear(hidden_layer,102),
                                  nn.LogSoftmax(dim = 1)  
                                ) 
        
        model.classifier = classifier
    
    return model    

def optim(model_name, model):

    if model_name == 'densenet_121':
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr = 0.01)
    else:
        optimizer = torch.optim.Adam(model.fc.parameters(), lr = 0.01)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)

    return optimizer, scheduler
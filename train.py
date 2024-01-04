import argparse
import torch
from torchvision import transforms, datasets
from torch import nn
#from torchsummary import summary
from classifier_model import build_model, optim
from tqdm import tqdm

parser = argparse.ArgumentParser()

def_model = 'resnet'
parser.add_argument('data_directory',
                        type = str,
                        default = '/Users/dalex/Udacity/aipnd-project/flower_data',
                        help = 'The path to the flowers dataset'
)

parser.add_argument('--save_dir',
                        type = str,
                        default = './checkpoint.pth',
                        help = 'The path where trained model will get saved'
)

parser.add_argument('--arch',
                        type = str,
                        default = 'resnet50',
                        help = 'Model architecture available :resnet50, densenet_121, defaulted to resnet50'
)

parser.add_argument('--hidden_units',
                    type = int,
                    default = 1000 if def_model =='resnet' else 512,
                    help = 
            'Hidden units for classifier for model selected, default values: 1000(resnet50), 512(densenet_121)'

)

parser.add_argument('--learning_rate',
                       type = float,
                       default = 0.001,
                       help = 'Learning rate to begin training the model with, default value 0.01'
                    )

parser.add_argument('--epochs',
                     type = int,
                     default= 5,
                     help = 'Epochs for conducting train and test loops'
                     )

parser.add_argument('--gpu',
                    type = str,
                    default= 'cuda',
                    help = 'Select whether to use GPU("gpu")/CPU("cpu") for training, default =gpu'
)

input = parser.parse_args()
data_dir = input.data_directory
path_to_save = input.save_dir
model_name = input.arch
hidden_layer = input.hidden_units
lr = input.learning_rate
epochs = input.epochs
gpu = input.gpu

#scheduler, optimizer
criterion = nn.NLLLoss()

#DATA PROCESSING

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([
                                      transforms.RandomResizedCrop(size= (224,224)),
                                      transforms.RandomHorizontalFlip(p= 0.5),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([transforms.RandomResizedCrop(size= (224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])    
])

img_datasets = {}
img_datasets['train'] = datasets.ImageFolder(train_dir, transform = train_transforms)
img_datasets['valid'] = datasets.ImageFolder(valid_dir, transform = test_transforms)
img_datasets['test'] = datasets.ImageFolder(test_dir, transform = test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(img_datasets['train'], batch_size = 32, shuffle = True)
validloaders = torch.utils.data.DataLoader(img_datasets['valid'], batch_size = 16, shuffle = True)
testloaders = torch.utils.data.DataLoader(img_datasets['test'], batch_size = 16, shuffle = True)


class_to_idx = img_datasets['train'].class_to_idx


#BUILDING MODEL ARCHITECTURE

new_model = build_model(model_name,hidden_layer)
optimizer, scheduler = optim(model_name, new_model,lr)

#TRAINING and VALIDATION
train_losses, valid_losses, test_losses, val_acc, test_acc = [],[],[],[],[]

def train_model(model,epochs,trainloaders, validloaders, criterion, optimizer, scheduler, gpu = 'gpu'):
    
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if gpu == 'gpu' else 'cpu'

    print('device is :', device)
    model.to(device)
    
    for e in range(epochs):
        running_train_loss = 0
    
        model.train()
        for images, labels in tqdm(trainloaders, desc=f'Epoch {e+1}/{epochs} - Training'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels) #NLLLoss
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        # Validation loop after training is complete
        running_val_accuracy_per_epoch = 0
        running_val_loss = 0
    
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(validloaders, desc=f'Epoch {e+1}/{epochs} - Validation'):
                images, labels = images.to(device), labels.to(device)

                log_ps = model(images)
                valid_loss = criterion(log_ps, labels)
                running_val_loss += valid_loss.item()
                prob_per_batch = torch.exp(log_ps)
                
                top_prob_per_batch, top_class_per_batch = prob_per_batch.topk(1, dim=1)
                truth_val = top_class_per_batch == labels.view(*top_class_per_batch.shape)
                running_val_accuracy_per_epoch += torch.mean(truth_val.type(torch.FloatTensor))

        epoch_val_accuracy = running_val_accuracy_per_epoch / len(validloaders)
        train_loss_per_epoch = running_train_loss / len(trainloaders)
        val_loss_per_epoch = running_val_loss / len(validloaders)

        scheduler.step(val_loss_per_epoch)
        
        train_losses.append(train_loss_per_epoch)
        valid_losses.append(val_loss_per_epoch)
        val_acc.append(epoch_val_accuracy)
        
        print(f'epoch: {e+1}/{epochs}')
        print(f'train_loss_per_epoch: {train_losses[-1]}')
        print(f'validation_loss_per_epoch: {valid_losses[-1]}')
        print(f'epoch_validation_accuracy: {epoch_val_accuracy.item() * 100}%')
        print(f'current learning rate: {optimizer.param_groups[0]["lr"]}')
        
    return model, train_losses, valid_losses, val_acc

trained_model, train_losses, valid_losses, val_acc = train_model(
    new_model, epochs, trainloaders, validloaders, criterion, optimizer, scheduler,'gpu') 

#TESTING MODEL ON TESTDATA

def test_model(model, epochs,testloaders, criterion, optimizer, scheduler, gpu = 'gpu'):
    
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if gpu == 'gpu' else 'cpu'
    
    print('device is :', device)
    model.to(device)
    
    for e in range(epochs):
        running_accuracy_per_epoch = 0
        running_test_loss = 0
        
        model.eval()
        for images, labels in tqdm(testloaders):
            
            images, labels = images.to(device), labels.to(device)
            prob_per_batch = torch.exp(model(images))
            test_loss = criterion(model(images), labels)
            running_test_loss += test_loss.item()
        
            top_prob_pre_batch ,top_class_per_batch = prob_per_batch.topk(1, dim =1)
            truth_val = top_class_per_batch == labels.view(*top_class_per_batch.shape)#this is bytes 
            running_accuracy_per_epoch += torch.mean(truth_val.type(torch.FloatTensor))
    
        epoch_test_accuracy = running_accuracy_per_epoch/len(testloaders) # divide by the number of batches to get average accuracy for the epoch
        test_loss_per_epoch = running_test_loss/len(testloaders)
        test_losses.append(test_loss_per_epoch)
        test_acc.append(epoch_test_accuracy)
    
        print(f'epoch: {e+1}/{epochs}')
        print(f'test_loss_per_epoch :{test_losses[-1]}')                     
        print(f'epoch_test_accuracy: {epoch_test_accuracy.item()*100}%')# changes single element tensor to a python number
        
    return test_losses, test_acc

test_losses, test_accuracy = test_model(
    trained_model, epochs, testloaders, criterion, optimizer, scheduler,'gpu' )

#SAVING CHECKPOINT, LOADING CHECKPOINT

checkpoint = {'input_size': 224,
              'output_size': 102,
              'model_name': model_name,
              'state_dict': trained_model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'scheduler_state_dict': scheduler.state_dict(),
              'class_to_idx': class_to_idx
             }

torch.save(checkpoint, path_to_save )

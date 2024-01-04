print("running predict.py")
import argparse
import torch

from classifier_model import build_model, optim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json


parser_next = argparse.ArgumentParser()

parser_next.add_argument('--path_to_image',
                               type = str,
                               default = '/Users/dalex/Udacity/aipnd-project/flower_data/test/6/image_07173.jpg',
                                help = 'Path to test image, default: /Users/dalex/Udacity/aipnd-project/flower_data/test/6/image_07173.jpg '
)

parser_next.add_argument('--top_k',
                               type = int,
                               default = 5,
                                help = 'select the number of top predictions required, default = 5'
)

parser_next.add_argument('--category_names',
                               type = str,
                               default = 'cat_to_name.json',
                                help = 'Path to labels dictionary(JSON file) containg mapping of class and flower names'
)

parser_next.add_argument('--checkpoint',
                               type = str,
                               default = './checkpoint.pth',
                                help = 'Path to checkpoint'
)

parser_next.add_argument('--gpu',
                               type = str,
                               default = 'gpu',
                                help = 'Perform inference using GPU(gpu) or CPU(cpu). Default:gpu'
)

input_args = parser_next.parse_args()

image_path = input_args.path_to_image
checkpt_path = input_args.checkpoint
device = input_args.gpu
topk = input_args.top_k
labels = input_args.category_names


#SETTING DEVICE
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if device == 'gpu' else 'cpu'

#LOADING THE CHECKPOINT 

def load_checkpoint(path):
    checkpoint = torch.load(path)
    name = checkpoint['model_name']
    model = build_model(name)
    optimizer, scheduler = optim(name,model)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

loaded_model = load_checkpoint(checkpt_path)

#IMAGE PROCESSING
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(image_path) as im:
        im = im.resize((256,256))
        mar_1 = 16 # (256-224)*0.5
        mar_2 = 240 #224+16
        im = im.crop((mar_1,mar_1,mar_2,mar_2))#l,u,r,l
        im = np.array(im)/255
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        norm_img = (im - mean)/ std
    
    return norm_img.transpose((2,0,1))
    
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image =image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

#PREDICTION

with open(labels, 'r') as f:
    cat_to_name = json.load(f)

idx_to_class={}#Extracting names and arranging as 'index: class' values instead of class:index
for  name, idx in loaded_model.class_to_idx.items():
    label = cat_to_name.get(name)  
    idx_to_class[idx] = label  

    
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print('prediction going on....')
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    #imshow(image)
    #plt.show()
    image = torch.FloatTensor(image)
    image.unsqueeze_(0) # batch dimension
    image = image.to(device)
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
    
    prob = torch.exp(output) 
    top_5_prob, top_5_class = prob.topk(5, dim=1)
    
    top_5_prob = top_5_prob.squeeze().tolist()
    top_5_class = top_5_class.squeeze().tolist()#indexes
    
    top_5_class_names = [idx_to_class[idx] for idx in top_5_class]
    
    return top_5_prob, top_5_class_names

Probs, classes= predict(image_path, loaded_model, topk)

print(f'Flower names of the top {topk} predictions are :', classes,
      ' with respective probabilities :', Probs)

truth = image_path.split('/')[-2]
print('Ground truth for the image is :',idx_to_class[loaded_model.class_to_idx[truth]])

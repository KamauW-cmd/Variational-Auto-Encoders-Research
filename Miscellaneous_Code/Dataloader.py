import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import math
import os
import torch.nn.functional as F
from skimage import io
from skimage.color import gray2rgb
import pandas as pd

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((200, 200), antialias=True)])

class Mugs_Datatset(Dataset):
   def __init__(self, csv_file, root_dir, transform=None):
       self.annotations = pd.read_excel(csv_file)
       self.root_dir = root_dir
       self.transform = transform
       datafile = pd.read_excel('/home/kamau/Final/mugsbb.xlsx')
       xy = datafile.to_numpy(dtype=np.float32) 
       self.x = torch.from_numpy(xy[:,0:])
       self.n_samples = xy.shape[0]

  
   def __len__(self):
       return(len(self.annotations))
  
   def __getitem__(self,index):
       img_path = os.path.join(self.root_dir, self.annotations.iloc[index,0])
       image = io.imread(img_path)
       y_label = self.x[index].clone().detach()


       if len(image.shape) == 2:
           image = gray2rgb(image)


       label = int(self.annotations.iloc[index,1])


       if self.transform:
           image = self.transform(image)


       return image, y_label

dataset = Mugs_Datatset(csv_file='label_maker.xlsx', root_dir='images', transform = transform)
train_loader = DataLoader(dataset=dataset, batch_size=4)

out = torch.empty(size = (0,5))
'''
for batch in train_loader:
    images,labels = batch
    for label in labels:
        if label[0] == 2:
            label = label.unsqueeze(0)
            out = torch.cat((out,label),0)
            print(label)
            print(out)
            break
    break

print(out)
'''
'''
for batch in train_loader:
    images,labels = batch
    print(labels[0].shape)
    print(labels[0])
    print(images[0].shape)
    old = labels[0].unsqueeze(0)
    new = torch.nn.functional.pad(old,(0,195,0,199))
    print(new.shape)
    break
'''
outputs = []
vessel = torch.empty(0,5)
handle = torch.empty(0,5)
cup = torch.empty(0,5)
index = []
vessel_img = torch.empty(0,200)
vessel_input = torch.empty(1,4,200,200)
track =0


'''
for batch in train_loader:
    images, labels = batch
    for i,label in enumerate(labels):
        if label[0] == 1:
            label = label.unsqueeze(0)
            vessel = torch.cat((vessel,label))
            index.append(i)
        elif label[0] == 0:
            label = label.unsqueeze(0)
            handle = torch.cat((handle,label))
        elif label[0] == 2:
            label = label.unsqueeze(0)
            cup = torch.cat((cup,label))
    
    for num in index:
        label = labels[num].unsqueeze(0)
        padded = torch.nn.functional.pad(label,(0,195,0,199))
        padded = padded.unsqueeze(0)
        #padded = F.pad(labels[num], (0,195,0,199))
        vessel_image = torch.cat((images[num],padded))
        vessel_image = vessel_image.unsqueeze(0)
        vessel_input = torch.cat((vessel_input,vessel_image))
    track += 1
    print(track)   

print(vessel_input.size())
    
'''     
val = 0
for batch in train_loader:
    val += 1

print(val)

   
            

'''

        if label[0].item() == 1:
            label = label.unsqueeze(0)
            vessel = torch.cat((vessel, label))
            index.append(i)
        if label[0].item() == 0:
            label = label.unsqueeze(0)
            handle = torch.cat((handle, label))
        if label[0].item() == 2:
            label = label.unsqueeze(0)
            cup = torch.cat((cup, label))
    
'''
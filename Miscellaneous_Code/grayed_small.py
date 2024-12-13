
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from Dataloader import Mugs_Datatset
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import math
import os
import time

torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6,max_split_size_mb:128'

num_classes = 3
batch_size = 2
num_epochs = 200
num_hidden = 200
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((200, 200), antialias=True)])

dataset = Mugs_Datatset(csv_file='label_maker.xlsx', root_dir='images', transform=transform)
train_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)

'''
nan = False
for batch in train_loader:
    images, labels = batch
    if torch.isnan(images).any():
        print("Nan Found")
        nan = True
        break
if not nan:
    print("No nan")



min = 0.5
max = 0.5

nan = False

for batch in train_loader:
    images, labels = batch
    if min>torch.min(images):
        min = torch.min(images)
    if max<torch.max(images):
        max = torch.max(images)
        
print(min)
print(max)
'''

class Reshape(nn.Module):
   def __init__(self, *args):
       super().__init__()
       self.shape = args

   def forward(self, x):
       return x.view(self.shape)

class Trim(nn.Module):
   def __init__(self, *args):
       super().__init__()

   def forward(self, x):
       return x[:,:,:200,:200]

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
                nn.Conv2d(4,16, stride=(1,1), kernel_size=(3,3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(16,32, stride=(2,2), kernel_size=(3,3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(32,64, stride=(2,2), kernel_size=(3,3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(64,128, stride=(1,1), kernel_size=(3,3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(128,256, stride=(1,1), kernel_size=(3,3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Flatten()
                
            )

        self.z_mean = torch.nn.Linear(256*50*50,200)
        self.z_log_var = torch.nn.Linear(256*50*50,200)

        self.decoder = nn.Sequential(
                torch.nn.Linear(200, 257*50*50),
                Reshape(-1, 257, 50, 50),
                
                nn.ConvTranspose2d(257, 128, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(128, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(32, 16, stride=(2, 2), kernel_size=(3, 3), padding=0),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(16, 4, stride=(1, 1), kernel_size=(3, 3), padding=0),
                Trim(),  # 1x29x29 -> 1x28x28
                nn.Sigmoid()
                )
        
    def projection(self,z,y):
        projected_label = self.label_projector(y.float())
        return z + projected_label

    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded, z_mean, z_log_var

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var / 2.)
        return z
    
    def check_tensor(self,x,layer_name):
        if torch.isnan(x).any():
            print(f"Nans detected in {layer_name}")
            
    def forward(self, x,y):
        out = self.encoder(x)
        encoded, z_mean, z_log_var = self.encoding_fn(x)
        encoded = F.sigmoid(encoded)
        self.check_tensor(encoded, "Encoded")
        decoded = self.decoding(encoded,y)
        self.check_tensor(decoded, "Decoded")
        return encoded, z_mean, z_log_var, decoded
    
    def decoding(self,x,y):
        padded = F.pad((y),(0,45,0,49))
        padded = padded.unsqueeze(0)
        encoded = torch.cat((x,padded))
        decoded = self.decoder(encoded)
        return(decoded)
    
    def sample(self, num_samples,y):
        with torch.no_grad():
            z = torch.randn(num_samples, num_hidden).to(device)
            samples = self.decoder(self.projection(z,y))
        return samples


def loss_function(x, decoded, z_mean, z_log_var):
   if torch.isnan(x).any(): print("NaNs detected in input images")
   if torch.isnan(z_mean).any(): print("NaNs detected in z_mean")
   if torch.isnan(z_log_var).any(): print("NaNs detected in z_log_var")
   if torch.isnan(decoded).any(): print("NaNs detected in decoded values")

   #assert torch.min(decoded) >= 0 and torch.max(decoded) <= 1, f"Decoded values are not in range Max Value: {torch.min(decoded)}, Min Value: {torch.max(decoded)}"
   #assert torch.min(x) >= 0 and torch.max(x) <= 1, "X values are not in range"

   reproduction_loss = torch.nn.functional.binary_cross_entropy(decoded, x, reduction='sum')
   KLD = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

   return reproduction_loss + KLD, reproduction_loss, KLD

def weights_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


model = VAE()
model.to(device)
model.apply(weights_init)
criterion = nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)


# Iterate over all batches
for epoch in range(num_epochs):
    #time1 = time.time()
    total_loss = 0
    for batch in train_loader:
        outputs = []
        vessel = torch.empty(0,5).to(device)
        handle = torch.empty(0,5).to(device)
        cup = torch.empty(0,5).to(device)
        index = []
        index2 = []
        vessel_img = torch.empty(0,200).to(device)
        vessel_input = torch.empty(0,0,200,200).to(device)

        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        for i,label in enumerate(labels):
            if label[0] == 1:
                label = label.unsqueeze(0).to(device)
                vessel = torch.cat((vessel,label)).to(device)
                index.append(i)
            elif label[0] == 0:
                label = label.unsqueeze(0).to(device)
                handle = torch.cat((handle,label)).to(device)
                index2.append(i)
            elif label[0] == 2:
                label = label.unsqueeze(0).to(device)
                cup = torch.cat((cup,label)).to(device)
        
        for num in index:
            label = labels[num].unsqueeze(0).to(device)
            padded = torch.nn.functional.pad(label,(0,195,0,199)).to(device)
            padded = padded.unsqueeze(0).to(device)
            #padded = F.pad(labels[num], (0,195,0,199))
            vessel_image = torch.cat((images[num],padded)).to(device)
            vessel_image = vessel_image.unsqueeze(0).to(device)
            vessel_input = torch.cat((vessel_input,vessel_image)).to(device)

        '''
        for num in index2:
            label = labels[num].unsqueeze(0).to(device)
            padded = torch.nn.functional.pad(label,(0,195,0,199)).to(device)
            padded = padded.unsqueeze(0).to(device)
            handle = padded.unsqueeze(0).to(device)
        '''


        handle= handle.to(device)
        vessel_input = vessel_input.to(device)
        #labels_one_hot = F.one_hot(handle,num_classes).to(device).long()
        encoded, mean, logvar, decoded = model(vessel_input)
        loss, rpl, kld = loss_function(vessel_img, decoded, mean, logvar)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(batch)

    epoch_loss = total_loss / len(train_loader)
    print(
        "Epoch {}/{}: loss={:.4f}".format(epoch + 1, num_epochs, epoch_loss)
    )
    outputs.append((epoch, images.cpu(), decoded.cpu().detach()))
    #time2 = time.time()
    #print(f"Time Elapsed: {time2-time1}")

#torch.cuda.empty_cache()

def display_images(epoch, valid, recon):
    num_images = valid.size(0)
    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))
    
    for i in range(num_images):
        # Check if the image has 3 channels or is grayscale
        if valid[i].shape[0] == 3:  # RGB image
            axes[0, i].imshow(valid[i].permute(1, 2, 0))
        else:  # Grayscale image
            axes[0, i].imshow(valid[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        
        if recon[i].shape[0] == 3:  # RGB image
            axes[1, i].imshow(recon[i].permute(1, 2, 0))
        else:  # Grayscale image
            axes[1, i].imshow(recon[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Recreated')
    
    plt.suptitle(f'Epoch {epoch+1}')
    plt.pause(0.1)
    plt.show()
    time.sleep(5)

# Display the images
for epoch, valid, recon in outputs:
    display_images(epoch, valid, recon)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from Dataloader import Mugs_Datatset
from torch.utils.data import DataLoader
from PIL import Image
torch.cuda.empty_cache()

batch_size = 10
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    
transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((830,965)), transforms.Grayscale()])

dataset = Mugs_Datatset(csv_file = 'label_maker.xlsx', root_dir = 'images', transform = transform)
train_loader = DataLoader(dataset= dataset, shuffle = True, batch_size = batch_size)

max_images = 0
min_images = 0

'''
for batch in train_loader:
    images,label = batch
    if torch.max(images)>max_images:
        max_images = torch.max(images)
    if torch.min(images)< min_images:
        min_images = torch.min(images)

print(max_images)
print(min_images)


'''
def weight_init(m):
    if isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias,0)


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self,x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:,:,:830,:965]


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
                nn.Conv2d(3,32, stride=(1,1), kernel_size=(3,3), padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.01, inplace = True),
                nn.Dropout2d(0.25),

                nn.Conv2d(32,64, stride=(2,2), kernel_size=(3,3), padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.01, inplace = True),
                nn.Dropout2d(0.25),

                nn.Conv2d(64,128, stride=(2,2), kernel_size=(3,3), padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.01, inplace = True),
                nn.Dropout2d(0.25),

                nn.Conv2d(128,256, stride=(1,1), kernel_size=(3,3), padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.01, inplace = True),
                nn.Dropout2d(0.25),

                nn.Conv2d(256,256, stride=(1,1), kernel_size=(3,3), padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.01, inplace = True),
                nn.Dropout2d(0.25),

                nn.Flatten(),
                
        )


        self.z_mean = torch.nn.Linear(12886016,200)
        self.z_log_var = torch.nn.Linear(12886016,200)

        self.decoder = nn.Sequential(
                torch.nn.Linear(200, 12886016),
                Reshape(-1, 256, 208, 242),

                nn.ConvTranspose2d(256, 256, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.01, inplace = True),
                nn.Dropout2d(0.25),

                nn.ConvTranspose2d(256, 128, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.01, inplace = True),
                nn.Dropout2d(0.25),


                nn.ConvTranspose2d(128, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.01, inplace = True),
                nn.Dropout2d(0.25),


                nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(0.01, inplace = True),
                nn.Dropout2d(0.25),

                nn.ConvTranspose2d(32, 3, stride=(1, 1), kernel_size=(3, 3), padding=0),
                Trim(), 
                nn.Sigmoid()
                )

    def encoding_fn(self,x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean,z_log_var)
        return encoded, z_mean, z_log_var

    def reparameterize(self,z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1))#.to(z_mu.get_device())
        z = z_mu+eps*torch.exp(z_log_var/2.)
        return z

    def forward(self,x):
        out = self.encoder(x)
        encoded, z_mean, z_log_var = self.encoding_fn(x)
        
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded

def loss_function(x, decoded, z_mean, x_log_var):

    if torch.isnan(x).any(): print("NaNs detected in input images") 
    if torch.isnan(z_mean).any(): print("NaNs detected in z_mean") 
    if torch.isnan(x_log_var).any(): print("NaNs detected in z_log_var")
    if torch.isnan(decoded).any(): print("NaNs detected in decoded values") 


    assert torch.min(decoded)>= 0 and torch.max(decoded)<=1, f"Decoded values are not in range Max Value: {torch.min(decoded)}, Min Value: {torch.max(decoded)}"
    assert torch.min(x)>= 0 and torch.max(x)<=1, "X values are not in range"
    reproduction_loss = torch.nn.functional.binary_cross_entropy(decoded,x, reduction = 'sum')
    KLD = -0.5*torch.sum(1+x_log_var-mean.pow(2)-x_log_var.exp())

    return reproduction_loss + KLD, reproduction_loss, KLD

model = VAE()
#model.to(device)
model.apply(weight_init)
criterion = nn.MSELoss(reduction = "sum")
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-5)

outputs = []

with torch.autograd.detect_anomaly(check_nan = True):
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            images,_ = batch
            images = images#.to(device)
            encoded,mean,logvar,decoded = model(images)
            loss,rpl,kld = loss_function(images,decoded,mean,logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*len(batch)
            
        #print(f'Loss: {loss} \n RPL: {rpl} \n KLD: {kld}')
        epoch_loss = total_loss/len(train_loader)
        #print(f'Epoch: {epoch+1}, Loss: {epoch_loss}:.4f}')
        
        print(
                "Epoch {}/{}: loss={:.4f}".format(epoch + 1, num_epochs, epoch_loss)
            )
        outputs.append((epoch, images.cpu(), decoded.cpu().detach()))

torch.cuda.empty_cache()


























'''
def display_images(epoch, valid, edited, recon):
    fig, axes = plt.subplots(3, 10, figsize=(15, 5))
    for i in range(10):
        axes[0, i].imshow(valid[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title('Original')
        axes[1, i].imshow(edited[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title('Edited')
        axes[2, i].imshow(recon[i].squeeze(), cmap='gray')
        axes[2, i].axis('off')
        axes[2, i].set_title('Recreated')
    plt.suptitle(f'Epoch {epoch+1}')
    plt.show()

for epoch, valid, edited, recon in outputs:
    display_images(epoch, valid, edited, recon)

'''
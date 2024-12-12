import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

batch_size = 64
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size = (27,27))
    ])


#Create Dataset
input_data = datasets.MNIST(root = './data', train=True, download = True, transform=transform)
valid_data = datasets.MNIST(root = './data', train=True, download = True, transform=transforms.ToTensor())


def randomize():
    #Generate Random Sequence
    rand_indx = torch.randperm(len(input_data))

    #Randomize Data
    in_shuffle_data = torch.utils.data.Subset(input_data,rand_indx)
    valid_shuffle_data = torch.utils.data.Subset(valid_data,rand_indx)

    #Create Dataloader
    input_loader = torch.utils.data.DataLoader(dataset=in_shuffle_data, batch_size = batch_size, shuffle = False)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_shuffle_data, batch_size = 64, shuffle = False)

    return input_loader, valid_loader

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
        return x[:,:,:28,:28]


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
                nn.Conv2d(1,32, stride=(1,1), kernel_size=(3,3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(32,64, stride=(2,2), kernel_size=(3,3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(64,128, stride=(2,2), kernel_size=(3,3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(128,256, stride=(1,1), kernel_size=(3,3), padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(256,256, stride=(1,1), kernel_size=(3,3), padding=1),
                nn.Flatten()
                
        )


        self.z_mean = torch.nn.Linear(12544,10)
        self.z_log_var = torch.nn.Linear(12544,10)

        self.decoder = nn.Sequential(
                torch.nn.Linear(10, 12544),
                Reshape(-1, 256, 7, 7),
                nn.ConvTranspose2d(256, 256, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(256, 128, stride=(1, 1), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(128, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
                nn.LeakyReLU(0.01),
                nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0),
                Trim(),  # 1x29x29 -> 1x28x28
                nn.Sigmoid()
                )

    def encoding_fn(self,x):
        x = self.encoder(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        encoded = self.reparameterize(z_mean,z_log_var)
        return encoded, z_mean, z_log_var

    def reparameterize(self,z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu+eps*torch.exp(z_log_var/2.)
        return z

    def forward(self,x):
        out = self.encoder(x)
        encoded, z_mean, z_log_var = self.encoding_fn(x)
        '''
        z_mean,z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        '''
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded

def loss_function(x, decoded, z_mean, x_log_var):
    reproduction_loss = torch.nn.functional.binary_cross_entropy(decoded,x, reduction = 'sum')
    KLD = -0.5*torch.sum(1+x_log_var-mean.pow(2)-x_log_var.exp())

    return reproduction_loss + KLD, reproduction_loss, KLD

model = VAE()
model.to(device)
criterion = nn.MSELoss(reduction = "sum")
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)

outputs = []


for epoch in range(num_epochs):
    total_loss = 0
    input_loader, valid_loader = randomize()
    iter_valid = iter(valid_loader)
    for batch in input_loader:
        images,_ = batch
        images = images.to(device)
        validation,_ = next(iter_valid)
        validation = validation.to(device)
        encoded,mean,logvar,decoded = model(images)
        loss,rpl,kld = loss_function(validation,decoded,mean,logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()*len(batch)
        
    #print(f'Loss: {loss} \n RPL: {rpl} \n KLD: {kld}')
    epoch_loss = total_loss/len(input_loader)
    #print(f'Epoch: {epoch+1}, Loss: {epoch_loss}:.4f}')
    
    print(
            "Epoch {}/{}: loss={:.4f}".format(epoch + 1, num_epochs, epoch_loss)
        )
    outputs.append((epoch, validation.cpu(), images.cpu(), decoded.cpu().detach()))


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
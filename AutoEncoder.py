import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
import matplotlib.pyplot as plt

batch_size = 64
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
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        #N, 1, 28,28
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16, 3, stride = 2, padding = 1), #N, 16,14,14
            nn.ReLU(),
            nn.Conv2d(16,32, 3, stride = 2, padding = 1), #N, 32,7,7
            nn.ReLU(),
            nn.Conv2d(32,64,7), #N 64,1,1

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64,32,7), #N 32,7,7
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,3, stride = 2, padding =1, output_padding = 1), #N 16,14,14
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,3, stride = 2, padding =1, output_padding = 1), #N 1,28,28
            nn.Sigmoid()
        )

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return (decoded)
    #Note: If our input is in the range of -1,1 then we would use nn.Tanh instead
    # of the sigmoid
    #nn.MaxPoo2D -> NN.MaxUnpool2D



model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)

num_epochs = 100
outputs = []
for epoch in range(num_epochs):
    input_loader, valid_loader = randomize()
    iter_valid = iter(valid_loader)
    for batch in input_loader:
        images,_ = batch
        images = images.to(device)
        validation,_ = next(iter_valid)
        validation = validation.to(device)
        recon = model(images)
        loss = criterion(recon,validation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')
    outputs.append((epoch, validation.cpu(), images.cpu(), recon.cpu().detach()))


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
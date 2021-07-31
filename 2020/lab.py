import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
BATCH_SIZE = 64

# Step1: Create Dataset and Dataloader
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])), 
    batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])), 
    batch_size=BATCH_SIZE, shuffle=True)

# Step2: Define custom CNN
class Net(nn.Module):
    def __init__(self):
        # create layers
        ## TODO ##

    def forward(self, x):
        # define forward propagation
        ## TODO ##

        return x

# Step3: Train the network
def train(num_epoch, model):   
    # create loss function and optimizer
    ## TODO ##

    # start training
    iteration = 0
    for epoch in range(num_epoch):
        for batch_idx, (images, labels) in enumerate(train_loader):
            ## TODO ##
            
            if iteration % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(images), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            iteration += 1

        # evaluate and save model
        eval_n_save(epoch, model)

# Step4: Valid the network
def eval_n_save(epoch, model):
    # evaluate model
    model.eval()
    total = 0
    correct = 0
    for images, labels in test_loader:
        ## TODO ##

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
    # save model
    os.makedirs('ckpt', exist_ok=True)
    filename = 'ckpt/%i_%.4f.pth'%(epoch,(correct / total))
    save_checkpoint(filename, model)
    
    return correct / total

# Step5: Save the model
def save_checkpoint(filename, model):
    # save model
    ## TODO ##

    print('model saved to %s\n'%filename)

    
if __name__ == '__main__':
    model = Net().cuda()
    train(10, model)
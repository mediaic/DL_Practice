# import library
## TODO ##



# define your oen model
class Onefc(nn.Module):
    def __init__(self):
        super(Onefc, self).__init__()
        self.fc = nn.Linear(32*32*3, 10)
    def forward(self, x):
        x = self.fc(x)
        return x
 
 
def train(num_epoch):
    # create dataset
    ## TODO ##
    
    # create model
    ## TODO ##
    
    # create loss function
    ## TODO ##
    
    # create optimizer
    ## TODO ##
    
    # start training
    for epoch in range(num_epoch):
        ## TODO ##
            
        
        # evaluate and save model
        
        
        
def eval(model, test_loader):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            outputs = model(images.view(images.shape[0],-1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
    return correct / total
    
    
if __name__ == '__main__':
    train(10)
    
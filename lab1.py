import torch
import torch.nn as nn
from torchvision import datasets, transforms

BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), 
    batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), 
    batch_size=BATCH_SIZE, shuffle=True)

model = nn.Sequential(
nn.Linear(32*32, 1024)
)

loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(10):
	loss_sum = 0
	for x_batch, y_batch in train_loader:
		x_batch = x_batch.view(x_batch.shape[0],-1)
		y_pred = model(x_batch)
		optimizer.zero_grad()
		loss_pre = loss(y_pred, y_batch)
		loss_sum = loss_sum+loss_pre
		loss_pre.backward()
		optimizer.step()
	print(loss_sum)
    
    # evaluate and save model
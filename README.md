Practice: Classification
===
In this practice, you will train a simple neural network classifier and play with many training tricks.

## Getting started

Before we start this practice, you need to understand how PyTorch framework (tensor, gradient, network, loss function, optimizer) works. Please refer to [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) and [examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html).

## Dataset
![CIFAR-10](https://storage.googleapis.com/kaggle-competitions/kaggle/3649/media/cifar-10.png)

The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

You can simply download the data with the torchvision API 
```python
from torchvision import datasets, transforms

BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])), 
    batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])), 
    batch_size=BATCH_SIZE, shuffle=True)
```

## Problem sets
Let’s train different models for recognizing CIFAR-10 classes! Please compare the convergence time and test accuracy by plotting the learning curves using tensorboard.

1. Build a softmax classification model with [a single linear layer](https://pytorch.org/docs/stable/nn.html#linear-layers) using [stochastic gradient descent (SGD)](https://pytorch.org/docs/stable/optim.html?highlight=gradient%20descent#torch.optim.SGD).  

2. Build a 1-hidden layer neural network with 1024 [ReLU units](https://pytorch.org/docs/stable/nn.html#relu) using SGD. This model should improve your test accuracy.

3. Try to get better performance by adding more layers and using [learning rate decay](https://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate).

4. Build a convolutional neural network with two [convolutional layers](https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d), followed by one fully connected layer using [Adam optimizer](https://pytorch.org/docs/stable/optim.html?highlight=gradient%20descent#torch.optim.Adam). (Both Conv1 and Conv2: 16@5x5 filters at stride 2)

5. Now, please replace the strides by a [max pooling](https://pytorch.org/docs/master/nn.html#maxpool2d) operation of stride 2, kernel size 2.

6. Apply [dropout](https://pytorch.org/docs/master/nn.html#dropout-layers) to the hidden layer of your models. Note that dropout should only be introduced during training, not evaluation.  

7. Load [ResNet18 pre-trained model](https://pytorch.org/docs/stable/torchvision/models.html#id3) and finetune on the CIFAR-10 dataset.

8. Train ResNet18 from scratch and compare the result to problem 7.

[***Optional***]

7. Apply [batch normalization](https://pytorch.org/docs/stable/nn.html?highlight=batchnorm#normalization-layers) to your models.

8. Replace the ReLU units in your models by [LeakyReLU](https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU) or [SELU](https://pytorch.org/docs/stable/nn.html#torch.nn.SELU).

## Saving your model
It is important to save your model at any time, especially when you want to reproduce your results or contiune the training procedure. One can easily save the model and the parapeters by using the [save/load functions](https://pytorch.org/docs/master/notes/serialization.html). While please also note that when you need to resume training, you should follow this [example](https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3) to save all required state dictionaries.

Now please save the model which achieves best performance among the above variants. Try to reproduce your results using the save/load functions instead of running a new training procedure.

##Bonus
If you have time, you can use the technique of transfer learning to achieve better performance of semantic segmentation.
Detailed discription is in [Segmentation Practice](https://github.com/mediaic/DL_Practice/tree/master/2018/2_Segmentation)
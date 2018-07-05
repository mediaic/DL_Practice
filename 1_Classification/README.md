Practice 1 : Classification
===
In this practice, you will train a simple neural network classifier and play with many training tricks.


## Dataset
![MNIST](https://www.tensorflow.org/images/mnist_0-9.png)
The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset comprises 60,000 training examples and 10,000 test examples of the handwritten digits 0–9, formatted as 28x28-pixel monochrome images.

You can simply download the data with the torchvision API
```
from torchvision import datasets, transforms

train_set = datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
                                      
test_set = datasets.MNIST('./data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
```

## Visualization

![tensorboard](https://www.tensorflow.org/images/mnist_tensorboard.png)

Please use tensorboard to visualize learning curves of different models in the following problem sets.  
Note that you need to additionaly install tensorboardX if using PyTorch.
```
pip3 install tensorboardX
```
See tutorials and examples of [tensorboardX](https://github.com/lanpa/tensorboard-pytorch).

## Problem sets

## Saving your model
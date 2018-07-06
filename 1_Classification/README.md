Practice 1 : Classification
===
In this practice, you will train a simple neural network classifier and play with many training tricks.

## Getting started

Before we start this practice, you need to understand how PyTorch framework (tensor, gradient, network, loss function, optimizer) works. Please refer to [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) and [examples](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html).

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
Let’s train different models for recognizing MNIST digits! Please compare the convergence time and test accuracy by plotting the learning curves using tensorboard.

1. Build a softmax regression model with [a single linear layer](https://pytorch.org/docs/stable/nn.html#linear-layers) using [stochastic gradient descent (SGD)](https://pytorch.org/docs/stable/optim.html?highlight=gradient%20descent#torch.optim.SGD).  

2. Build a 1-hidden layer neural network with 1024 [ReLU units](https://pytorch.org/docs/stable/nn.html#relu) using SGD. This model should improve your test accuracy.

3. Try to get better performance by adding more layers and using [learning rate decay](https://pytorch.org/docs/master/optim.html#how-to-adjust-learning-rate).

4. Build a convolutional neural network with two [convolutional layers](https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d), followed by one fully connected layer using [Adam optimizer](https://pytorch.org/docs/stable/optim.html?highlight=gradient%20descent#torch.optim.Adam). (Both Conv1 and Conv2: 16@5x5 filters at stride 2)

5. Now, please replace the strides by a [max pooling](https://pytorch.org/docs/master/nn.html#maxpool2d) operation of stride 2, kernel size 2.

6. Apply [dropout](https://pytorch.org/docs/master/nn.html#dropout-layers) to the hidden layer of your models. Note that dropout should only be introduced during training, not evaluation.  

[***Optional***]

7. Apply [batch normalization](https://pytorch.org/docs/stable/nn.html?highlight=batchnorm#normalization-layers) to your models.

8. Replace the ReLU units in your models by [LeakyReLU](https://pytorch.org/docs/stable/nn.html#torch.nn.LeakyReLU) or [SELU](https://pytorch.org/docs/stable/nn.html#torch.nn.SELU).

## Saving your model
It is important to save your model at any time, especially when you want to reproduce your results or contiune the training procedure. One can easily save the model and the parapeters by using the [save/load functions](https://pytorch.org/docs/master/notes/serialization.html). While please also note that when you need to resume training, you should follow this [example](https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3) to save all required state dictionaries.

Now please save the model which achieves best performance among the above variants. Try to reproduce your results using the save/load functions instead of running a new training procedure.
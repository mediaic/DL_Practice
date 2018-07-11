Practice 2 : Segmentation
===
In this practice, you will use the technique of transfer learning to achieve better performance of semantic segmentation.

## Task Definition
Semantic segmentation predicts a label to each pixel with CNN models.

![task](https://i.imgur.com/feD4Hv2.png)

* Input : RGB image
* Output : pixel-wise label prediction


| Object | Color | Label ID |
| -------- | -------- | -------- |
| Urban land | Cyan: (0,255,255) | 0    |
| Agriculture land | Yellow: (255,255,0) | 1    |
| Rangeland | Purple: (255,0,255) | 2    |
| Forest land  | Green: (0,255,0) | 3    |
| Water | Blue: (0,0,255) | 4    |
| Barren land | White: (255,255,255) | 5    |
| Unknown | Black: (0,0,0) | 6    |

## Dataset Description
```
seg_data.zip
 ⊢ training/
 ⊢ validation/
```

[Download](https://drive.google.com/file/d/11UOONxz2djoeKbEGeh7ihxw3W3YCoTrq/view?usp=sharing)  
**Please DO NOT distribute the data for other purposes.**

### Training set
* Contains 9252 256x256 image-mask (ground truth) pairs
* Satellite images are named *'xxxx_sat.jpg'*
* Mask images (ground truth) are named *'xxxx_mask.png'*

### Validation set
* Contains 1028 256x256 image-mask pair
* Naming rules are the same as the training set
* You **CANNOT** use validation data for training purposes.

## Implementation

### Fully Convolutional Network (FCN)
![FCN32s](https://i.imgur.com/x8PX7ZX.png)

### Loading pre-trained VGG16 weights
![VGG](https://i.imgur.com/RQadXlb.png)

The parameters of above layers can be initialize with VGG16 weights trained on ImageNet. Detailed network settings can be found [here](https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/models/fcn.py).  
You can also directly extract block5 features using torchvision functions.

```python
from torchvision import models

vgg = models.vgg16(pretrained=True)
features5 = nn.Sequential(*list(vgg.features.children()))
```

## Problem Sets
Please compare the convergence time and accuracy by plotting the learning curves using tensorboard.

1. Train a FCN 32s model initialized with the parameters pre-trained on ImageNet.

2. Train a FCN 32s model from draft. (i.e., you cannot use pre-trained weights. Please note that it might be ***VERY DIFFICULT*** to get a preferable performance. Do not stuck on this problem since it is just for comparison.)


[***optional***]

3. Design an improved model which performs better than above models. Here are some good [examples](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html) for semantic segmentation.


## Evaluation Metric

### Intersection over Union (IoU)
For each class, the IoU is defined as: 
> *True Positive / (True Positive + False Positive + False Negative)*

To be more specific, it can be calculated as the ratio of the **overlap area** to the **union area** of ground-truth and prediction.

<img src="https://i.imgur.com/zhCdhwG.png" height="350">

### Mean Intersection over Union (mIoU)
The mIoU is calculated by averaging over all classes except for **Unknown(0,0,0)**. Please note that mIoU need to be calculated over all validation images not a single one.

## Baseline
Can you beat the baseline of validation set, mIoU = **0.635**?
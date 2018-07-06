Practice 2 : Segmentation
===
In this practice, you will use the technique of transfer learning to achieve better performance of semantic segmentation.

## Task Definition
Semantic segmentation predicts a label to each pixel with CNN models.

![task](https://i.imgur.com/feD4Hv2.png)

* Input : RGB image
* Output : pixel-wise label prediction

## Dataset Description
```
seg_data.zip
 ⊢ training/
 ⊢ validation/
```
### training set
* Contains 2313 image-mask (ground truth) pairs
* Satellite images are named *'xxxx_sat.jpg'*
* Mask images (ground truth) are named *'xxxx_mask.png'*

### validation set
* Contains 257 image-mask pair
* Naming rules are the same as the training set
* You **CANNOT** use validation data for training purposes.

# Implementation



# Evaluation Metric

### Intersection over Union (IoU)
For each class, the IoU is defined as: 
> *True Positive / (True Positive + False Positive + False Negative)*

To be more specific, it can be calculated as the ratio of the **overlap area** to the **union area** of ground-truth and prediction.

![IOU](https://i.imgur.com/zhCdhwG.png =350x)

### mean Intersection over Union (mIoU)
The mIoU is calculated by averaging over all classes except for **Unknown(0,0,0)**. Please note that mIoU need to be calculated over all validation images not a single one.

# Baseline
Can you beat the baseline of validation set, mIoU = **0.635**?
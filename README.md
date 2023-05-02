# Normalization-Test
Comparison normalization when using a pre-trained model

This experiment aims to determine whether it is better to use the mean and std of the normalization when using pertrained weights for the pretrained dataset or the fine-tuning dataset.

**Motivation**

In general, it is known that normalization is used to fit the data for fine-tuning. This is because the data for fine-tuning is different from the distribution of the data used for pre-training. However, on the other hand, since the weights of the pre-trained model are trained according to the distribution of the pre-training data, I wondered if it would be appropriate to apply the same normalization to the data for fine-tuning as was used in the pre-training data.

# Experiments

**Baseline**

In this experiments, I used different types of CNN models, ResNet and DenseNet, and varied the size of the models to use them as baseline models.

| model       |             #Params |
|:------------|--------------------:|
| ResNet18    |                 11M |
| ResNet50    |                 24M |
| DenseNet121 |                  7M |
| DenseNet161 |                 27M |


**Dataset**

- CIFAR10
- CIFAR100

**Results**


Figure 1 shows the accuracy history of the testset between different models and datasets. From the results, it seems that using normalization on the fine-tuning dataset converges to higher performance in most cases.

<p align='cetner'>
    <img src="https://github.com/TooTouch/Normalization-Test/blob/main/assets/figure1.jpg?raw=true">
    <strong>Figure 1</strong>. Compare history of test accuracy based on normalization settings by model and dataset.
</p>



Table 1 compares the best accuracy for the testset. When comparing the results quantitatively, we found that using normalization based on the finetuning dataset resulted in higher performance in three out of four cases for CIFAR10 and higher performance in two out of four cases for CIFAR100.



**Table 1**. Test accuracy based on normalization settings by model and dataset.
| Model           | Setting           |   Accuracy (CIFAR10) |   Accuracy (CIFAR100) |
|:----------------|:------------------|---------------------:|----------------------:|
| DenseNet121     | Finetune          |           **0.8901** |                0.6336 |
|                 | Pretrained        |               0.8873 |            **0.6422** |
| DenseNet161     | Finetune          |               0.9085 |            **0.6857** |
|                 | Pretrained        |           **0.9089** |                0.6807 |
| ResNet18        | Finetune          |           **0.8847** |            **0.6164** |
|                 | Pretrained        |               0.8797 |                0.6085 |
| ResNet50        | Finetune          |           **0.8951** |                0.6696 |
|                 | Pretrained        |               0.895  |            **0.6739** |


# Conclusion

The experimental results show that using normalization based on the finetuning datasets is often better, but it is difficult to say exactly how good it is. In the future, I would like to compare more models and data to generalize the results.
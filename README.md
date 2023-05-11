# Normalization-Test
Comparison normalization to pre-process images when using a pre-trained model

**Motivation**

It is generally known that when using a pre-trained model for fine-tuning, the image for fine-tuning is normalized using the statistics of the data to be used for fine-tuning. This is because the data for fine-tuning is different from the distribution of the data used for pre-training. However, sometimes the data to be fine-tuned is normalized by the statistics of the data used for pre-training, and the performance is better or similar. Therefore, I would like to experiment to see in which cases this difference occurs.

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

We use four datasets for our experiments. We also use the mean and standard deviation to check the statistics after normalization, or the mean and standard deviation of ImageNet-1k used for model pre-training.


< **Table 1**. Dataset Description.  >
| Dataset           | #Images(trainset) | #Images(testset) | #Classes | Image Size  | Fine-tune stats.                                   | ImageNet-1k stats.                                    |
|-------------------|-------------------|------------------|----------|-------------|---------------------------------------------------|-------------------------------------------------------|
| CIFAR10           | 50,000            | 10,000           | 10       | 32 X 32 X 3 | mean: (0.00, 0.00, 0.00)<br>std : (1.00, 1.00, 1.00) | mean: (0.03, 0.12, 0.18)<br>std : (1.08, 1.09, 1.16)     |
| CIFAR100          | 50,000            | 10,000           | 100      | 32 X 32 X 3 | mean: (0.00, 0.00, 0.00)<br>std : (1.00, 1.00, 1.00) | mean: (0.10, 0.14, 0.16)<br>std : (1.17, 1.15, 1.23)     |
| SVHN              | 73,257            | 26,032           | 10       | 32 X 32 X 3 | mean: (0.00, 0.00, 0.00)<br>std : (1.00, 1.00, 1.00) | mean: (-0.21, -0.06, 0.30)<br>std : ( 0.87,  0.90, 0.88) |
| Tiny ImageNet-200 | 100,000           | 10,000           | 200      | 64 X 64 X 3 | mean: (0.00, 0.00, 0.00)<br>std : (1.00, 1.00, 1.00) | mean: (-0.38, -0.04, 0.33)<br>std : ( 1.23,  1.20, 1.23) |


**Training Settings**

All models are trained with `./default_configs`.

```
SEED: 223

DATASET:
  datadir: /datasets

OPTIMIZER:
  opt_name: Adam
  lr: 0.001

TRAINING:
  batch_size: 256
  test_batch_size: 256
  epochs: 100
  log_interval: 10
  use_scheduler: true

RESULT:
  savedir: './saved_model'
```

the statistics for each datasets to be fine-tuned and ImageNet-1k used to pre-trained can be find in `./stats.py`.


```
dataset_stats = {
    "imagenet":{
        "num_classes" : 1000,
        "img_size"    : 224,
        "mean"        : (0.485, 0.456, 0.406),
        "std"         : (0.229, 0.224, 0.225)
    },
    "cifar10":{
        "num_classes" : 10,
        "img_size"    : 32,
        "mean"        : (0.4914, 0.4822, 0.4465),
        "std"         : (0.247, 0.2435, 0.2616)
    },
    "cifar100":{
        "num_classes" : 100,
        "img_size"    : 32,
        "mean"        : (0.5071, 0.4867, 0.4408),
        "std"         : (0.2675, 0.2565, 0.2761)
    },
    "svhn":{
        "num_classes" : 10,
        "img_size"    : 32,
        "mean"        : (0.4377, 0.4438, 0.4728), 
        "std"         : (0.1980, 0.2010, 0.1970)
    },
    "tiny_imagenet_200":{
        "num_classes" : 200,
        "img_size"    : 64,
        "mean"        : (0.4802, 0.4481, 0.3975), 
        "std"         : (0.2764, 0.2689, 0.2816)
    }
}
```

**Results**


<p align='cetner'>
    <img src="https://github.com/TooTouch/Normalization-Test/blob/main/assets/figure1.jpg?raw=true">
    < <strong>Figure 1</strong>. Compare history of test accuracy based on normalization settings by model and dataset >
</p>


Figure 1 shows the accuracy history of the testset between different models and datasets. From the results, it seems that using normalization on the fine-tuning dataset converges to higher performance in most cases.


< **Table 2**. Accuracy(%) for four datasets. $\textcolor{green}{green}$ indicates how much better the performance is than pre-trained setting. $\textcolor{red}{red}$ indicates how much worse the performance is than pre-trained setting. >

| Model           | Setting           |   CIFAR10<br>Accuracy(%) |  CIFAR100<br>Accuracy(%) |      SVHN<br>Accuracy(%) |       Tiny ImageNet-200<br>Accuracy(%) |
|:----------------|:------------------|---------------------:|----------------------:|---------------------:|-------------------------------:|
| DenseNet121     | Fine-tuned         |   **89.05** ($\textcolor{green}{+00.02}$) |        63.34 ($\textcolor{red}{-00.24}$) |    **95.16** ($\textcolor{green}{+00.25}$)|              **59.66** ($\textcolor{green}{+00.42}$)|
|                 | Pre-trained        |                89.03 |             **63.58** |                94.91 |                          59.24 |
|                 |                   |                      |                       |                      |                                |
| DenseNet161     | Fine-tuned         |         90.91 ($\textcolor{red}{-00.15}$) |        67.68 ($\textcolor{red}{-00.19}$) |    **95.51** ($\textcolor{green}{+00.05}$)|              **61.90** ($\textcolor{green}{+00.03}$)|
|                 | Pre-trained        |            **91.06** |             **67.87** |                95.46 |                          61.87 |
|                 |                   |                      |                       |                      |                                |
| ResNet18        | Fine-tuned         |         87.88 ($\textcolor{red}{-00.10}$) |    **61.67** ($\textcolor{green}{+00.12}$) |        95.01 ($\textcolor{red}{-00.16}$)|              **53.62** ($\textcolor{green}{+00.73}$)|
|                 | Pre-trained        |            **87.98** |                 61.55 |            **95.17** |                          52.89 |
|                 |                   |                      |                       |                      |                                |
| ResNet50        | Fine-tuned         |         89.78 ($\textcolor{red}{-00.42}$) |    **67.58** ($\textcolor{green}{+00.64}$) |        94.39 ($\textcolor{red}{-00.09}$)|              **67.32** ($\textcolor{green}{+00.16}$)|
|                 | Pre-trained        |            **90.20** |                 66.94 |            **94.48** |                          67.16 |


Table 1 compares the best accuracy for the testset. When comparing the results quantitatively, I found that using normalization based on the fine-tuning dataset resulted in higher performance in three out of four cases for CIFAR10 and higher performance in two out of four cases for CIFAR100 and SVHN. However, Tiny ImageNet-200 performed better with all of the fine-tuned settings.

In general, image normalization for model training is done to fit a normal distribution. Therefore, the pre-training model was also trained on images normalized to a normal distribution. However, in the case of tiny imagenet-200, when normalized to the statistics of imagenet-1k, it is farther from the normal distribution than the other datasets, so I assume that the fine-tuning setting yielded better results than the pretraining setting.


< **Table 3**. Accuracy(%) for four datasets and for normalization settings.
| Model           | Setting           |   CIFAR10<br>Accuracy(%) |   CIFAR100<br>Accuracy(%) |   SVHN<br>Accuracy(%) |   Tiny ImageNet-200<br>Accuracy(%) |
|:----------------|:------------------|-------------------------:|--------------------------:|----------------------:|-----------------------------------:|
| DenseNet121     | finetune          |               **0.8905** |                    0.6334 |            **0.9516** |                         **0.5966** |
|                 | pretrain          |            <u>0.8903</u> |             <u>0.6358</u> |                0.9491 |                      <u>0.5924</u> |
|                 | instance          |                   0.8768 |                    0.6056 |         <u>0.9515</u> |                             0.5659 |
|                 | minmax            |                   0.8889 |                **0.6367** |                0.9512 |                             0.5917 |
|                 |                   |                          |                           |                       |                                    |
| DenseNet161     | finetune          |            <u>0.9091</u> |             <u>0.6768</u> |            **0.9551** |                      <u>0.6190</u> |
|                 | pretrain          |               **0.9106** |                **0.6787** |         <u>0.9546</u> |                             0.6187 |
|                 | instance          |                   0.8994 |                    0.6572 |                0.9537 |                             0.6034 |
|                 | minmax            |                   0.9087 |                    0.6763 |                0.9525 |                         **0.6245** |
|                 |                   |                          |                           |                       |                                    |
| ResNet18        | finetune          |                   0.8788 |                **0.6167** |         <u>0.9501</u> |                         **0.5362** |
|                 | pretrain          |               **0.8798** |             <u>0.6155</u> |            **0.9517** |                      <u>0.5289</u> |
|                 | instance          |                   0.8655 |                    0.5764 |                0.9455 |                             0.5083 |
|                 | minmax            |            <u>0.8797</u> |                    0.6118 |                0.9497 |                             0.5254 |
|                 |                   |                          |                           |                       |                                    |
| ResNet50        | finetune          |                   0.8978 |                **0.6758** |                0.9439 |                         **0.6732** |
|                 | pretrain          |            <u>0.9020</u> |                    0.6694 |                0.9448 |                             0.6716 |
|                 | instance          |                   0.8932 |                    0.6469 |            **0.9465** |                             0.6540 |
|                 | minmax            |               **0.9023** |             <u>0.6750</u> |         <u>0.9449</u> |                      <u>0.6724</u> |

Following on from Table 2, we experimented with two additional image normalization methods in Table 3. The first instance setting performs normalization on a per-image basis, and the second performs MinMax normalization. The results were surprisingly good with minmax normalization in some cases.


# Conclusion

The results were, unsurprisingly, strongly influenced by whether the image was normalized to a normal distribution. The general rule of thumb is to normalize to the distribution of the data we want to fine-tune. Sometimes I do it without bothering with normalization, and I see it in other repositories as well, so I need to reflect on it and make sure the model is trained well according to the canonicals.





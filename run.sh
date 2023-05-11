model_list="resnet18 resnet50 densenet121 densenet161"
datasets="CIFAR10 CIFAR100 SVHN tiny_imagenet_200"

for d in $datasets
do
    for m in $model_list
    do
        python main.py --default_setting default_configs.yaml --dataname $d --modelname $m --normalize finetune
        python main.py --default_setting default_configs.yaml --dataname $d --modelname $m --normalize pretrain
        python main.py --default_setting default_configs.yaml --dataname $d --modelname $m --normalize instance
        python main.py --default_setting default_configs.yaml --dataname $d --modelname $m --normalize minmax
    done
done
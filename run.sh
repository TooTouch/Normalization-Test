model_list="resnet18 resnet50 densenet121 densenet161"
datasets="CIFAR10 CIFAR100"

for d in $datasets
do
    for m in $model_list
    do
        python main.py --yaml_config ./configs/$d/$m-finetune_set.yaml
        python main.py --yaml_config ./configs/$d/$m-pretrained_set.yaml
    done
done
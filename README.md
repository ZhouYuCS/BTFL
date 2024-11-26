# How to Run
This repostory mainly inherits from [`https://github.com/lins-lab/fedthe`](https://github.com/lins-lab/fedthe), please check this link for environments construction.
The main entry point of a single experiment is [`main.py`](main.py). To facilitate experiments running, we provide [`scripts`](exps/) for running the bulk experiments in the paper. For example, to run `BTFL` and other baselines on CIFAR10 with CNN, you can run the following command:
```
python run_exps.py --script_path exps/exp_cifar10_cnn.py
```
For other architectures, simply do:
```
python run_exps.py --script_path exps/exp_cifar10_cct.py
python run_exps.py --script_path exps/exp_cifar10_resnet.py
```
For running on ImageNet with ResNet, you can run the following:
```
python run_exps.py --script_path exps/exp_imagenet_resnet.py
```
For running on OfficeHome with ResNet, you can run the following:
```
python run_exps.py --script_path exps/exp_oh_resnet.py
```


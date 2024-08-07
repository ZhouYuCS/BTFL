# How to Run
The main entry point of a single experiment is [`main.py`](main.py). To facilitate experiments running, we provide [`scripts`](exps/) for running the bulk experiments in the paper. For example, to run `BTFL` and other baselines on CIFAR10 with CNN (Tab. 1), you can run the following command:
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


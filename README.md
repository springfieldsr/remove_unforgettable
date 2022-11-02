## Execution

```bash
usage: image_cls.py [-h] [--datasets {CIFAR10,CIFAR100}] [--models {resnet18, resnet101}]\
                              [--bs {1,4,16,32,64,256,1024}] [--epochs EPOCHS] [--lr LR]\
                              [--streak streak] [--es early_stop] [--check_interval]\
                              [--baseline]

optional arguments:
  -h, --help            show this help message and exit
  --dataset {CIFAR10,CIFAR100} datasets
  --model {resnet*}     models
  --bs {1,4,16,64,256,1024}
                        batch_size
  --epochs EPOCHS       epochs of training
  --lr LR               learning rate
  --streak              memorization streak number for an example to be removed
  --check_interval      epoch interval to evaluate the memorized data
  --es                  early stop or not
  --baseline            train without any modification
```
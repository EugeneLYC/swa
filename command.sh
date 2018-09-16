#! /bin/bash
python3 train.py --dir=runs --dataset=CIFAR10 --data_path=. --model=VGG16 --epochs=300 --lr_init=0.05 --wd=5e-4 --swa --swa_start=1 --swa_lr=0.01 --swa_c_epochs=5

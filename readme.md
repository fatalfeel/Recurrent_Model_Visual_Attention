# Reccurrent Visual Attention Model

## Introduction
This repo is an implementation of Reccurrent Attention Model from [Recurrent Models of Visual Attention](http://papers.nips.cc/paper/5542-recurrent-models-of-visual-attention.pdf). It with a lot of comments and set image show let you easier to understand.

Using glimpses 6 after traning 100 epochs the best validation accuracy is 99.12%

## Requirements

- Python 3.6+
- PyTorch 0.4

## Usage

- Step debug on a CPU device, Cuda can not.
- python3 ./train.py

## Others can help you learn more

- https://www.mediafire.com/file/rb3f42h0jbord11/recurrent-visual-attention.tar.xz
- python3 ./main.py --use_gpu False --is_train True --num_patches 2 --glimpse_scale 2

- https://www.mediafire.com/file/kl7epkvxrg1641n/Visual-Attention-Pytorch.tar.xz
- python3 ./main.py --n_glimpses 6

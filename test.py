from __future__ import print_function
import argparse
import os

import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
from torch.nn import functional as tnf
from torchvision import datasets, transforms
from torchvision.utils import save_image

# im = imread('5.png', mode='F')
# size = im.shape[0]
# im = im.reshape(1, 1, size, size) / 255.0
# print(im.shape)

# gsize = 600
# factor = gsize / size
# # gsize = int(size*factor)

# a = F.affine_grid(Variable(torch.tensor([[[0.8,0,0], [0,0.8,0.0]]])), torch.Size([1,1,gsize,gsize]))
# # a = Variable(torch.randn(1,5,5,2))
# print(a.size())
# x = Variable(torch.tensor(im))
# s = F.grid_sample(x, a)
# print(s.size())
# save_image(x, 'x.png')
# save_image(s, 's.png')
# exit()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser(description='VAE MNIST Example')

parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 128)')

parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')

#default=False for step debug
#parser.add_argument('--no-cuda', action='store_true', default=True,
parser.add_argument('--cuda', type=str2bool, default=False,
                    help='enables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--test-t-e-s-t', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda:0" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

def show_MNIST(self, img):
    grid = torchvision.utils.make_grid(img)
    trimg = grid.numpy().transpose(1, 2, 0)
    '''cv2.imshow('MNIST', trimg)
    cv2.waitKey(2000);
    cv2.destroyAllWindows()'''
    plt.imshow(trimg)
    plt.title('Batch from dataloader')
    plt.axis('off')
    plt.show()

def Retina(data, location, output_size, nsc):
    # Transform image to retina representation
    # Assume that width = height and channel = 1

    batch_size, input_size = data.size(0), data.size(2) - 1
    # assert output_size * 2**(nsc - 1) <= input_size, "output_size * 2**(nsc-1) should smaller than or equal to input_size"

    # construct theta for affine transformation
    theta = torch.zeros(batch_size, 2, 3)
    theta[:, :, 2] = location

    scale = output_size / input_size
    originsize = torch.Size([batch_size, 1, output_size, output_size])

    output = torch.zeros(batch_size, output_size * output_size * nsc)

    for i in range(nsc):
        theta[:, 0, 0] = scale
        theta[:, 1, 1] = scale
        grid = tnf.affine_grid(theta, originsize, align_corners=True)  # theta is location shift

        # glimpse = tnf.grid_sample(data, grid).view(batch_size, -1)
        sample = tnf.grid_sample(data, grid, align_corners=True)
        # show_MNIST(sample)
        glimpse = sample.view(batch_size, -1)

        #output[:, i * output_size * output_size: (i + 1) * output_size * output_size] = glimpse
        fillsize    = output_size * output_size
        pos_start   =  i        * fillsize
        pos_end     = (i + 1)   * fillsize
        output[:, pos_start: pos_end] = glimpse
        scale *= 2.0

    return output.detach()

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../../data',
                                                              train=True,
                                                              download=True,
                                                              transform=transforms.ToTensor()),
                                                              batch_size=args.batch_size,
                                                              shuffle=True,
                                                              **kwargs)

    location = torch.empty(args.batch_size, 2)
    location[:, 0] = 0.5
    location[:, 1] = -0.5

    for batch_idx, (data, label) in enumerate(train_loader):
        data = data.to(device)
        output = Retina(data, location, 10, 2)
        break

    print(label.size())
    print(output.size())
    os.makedirs('test', 0o777, exist_ok=True)
    save_image(data, './test/data.png')
    save_image(output.view(args.batch_size, 1, 20, 10), './test/output.png')

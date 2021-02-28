import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as tnf
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from modelvt import ModelVT
from utils import draw_locations

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

parser = argparse.ArgumentParser(description='Args of Train')

parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size for training (default: 20)')

parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='init learning rate (default: 1e-3)')

parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')

#parser.add_argument('--no-cuda', action='store_true', default=True,
#default=False for step debug
parser.add_argument('--cuda', type=str2bool, default=False,
                    help='enables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--input-size', type=int, default=28, metavar='N',
                    help='input image size for training (default: 28)')

parser.add_argument('--location-size', type=int, default=2, metavar='N',
                    help='input location size for training (default: 2)')

parser.add_argument('--location-std', type=float, default=0.15, metavar='N',
                    help='standard deviation used by location network (default: 0.15)')

parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='input action size (number of classes) for training (default: 10)')

parser.add_argument('--glimpse-size', type=int, default=8, metavar='N',
                    help='glimpse image size for training (default: 8)')

parser.add_argument('--num-glimpses', type=int, default=6, metavar='N',
                    help='number of glimpses for training (default: 6)')

parser.add_argument('--num-scales', type=int, default=2, metavar='N',
                    help='number of scales (retina patch) for training (default: 2)')

parser.add_argument('--feature-size', type=int, default=128, metavar='N',
                    help='location and input glimpse feature size for training (default: 128)')

parser.add_argument('--glimpse-feature-size', type=int, default=256, metavar='N',
                    help='output glimpse feature size for training (default: 256)')

parser.add_argument('--hidden-size', type=int, default=256, metavar='N',
                    help='feature size for RNN (default: 256)')

args = parser.parse_args()

assert args.glimpse_size * 2**(args.num_scales - 1) <= args.input_size, "glimpse_size * 2**(num_scales-1) should smaller than or equal to input-size"
torch.manual_seed(args.seed)
device = torch.device("cuda:0" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

#practice (h.) NLL-Loss & CrossEntropyLoss - nll_crossEntropy.py
def CalculateLoss(labels, act_probs, location_log_probs, critic_values, celoss_fn):
    predictions     = torch.argmax(act_probs, dim=1, keepdim=True) #return the index of max number in a row
    action_loss     = celoss_fn(act_probs, labels.squeeze())  # CrossEntropyLoss

    num_repeats     = critic_values.size(-1)
    rewards         = (predictions == labels).detach().float().repeat(1, num_repeats)
    #baseline_loss  = tnf.mse_loss(rewards, critic_values)
    baseline_loss   = tnf.mse_loss(critic_values, rewards)  #in ppo its mean value_loss

    rv_difference   = rewards - critic_values.detach()
    advantages      = (rv_difference - rv_difference.mean()) / (rv_difference.std() + 1e-5)
    #reinforce_loss  = torch.mean(torch.sum(-location_log_probs * rv_difference, dim=1))
    reinforce_loss  = torch.sum(-location_log_probs * advantages, dim=1)

    return action_loss + baseline_loss + reinforce_loss

def train(modelRAM, epoch, train_loader, celoss_fn):
    modelRAM.train()
    #train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data        = data.to(device)
        labels      = labels.to(device)
        optimizer.zero_grad()
        act_probs, _, location_log_probs, critic_values = modelRAM(data)
        labels      = labels.unsqueeze(dim=1)
        loss        = CalculateLoss(labels, act_probs, location_log_probs, critic_values, celoss_fn)
        loss.mean().backward()
        #train_loss += loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t'.format(epoch,
                                                               batch_idx * len(data),
                                                               train_size,
                                                               100. * batch_idx / len(train_loader)))
    #print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / train_size))

def test(modelRAM, epoch, data_source, size):
    modelRAM.eval()
    total_correct = 0.0
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(data_source):
            data    = data.to(device)
            labels  = labels.to(device)
            act_probs, _, _, _ = modelRAM(data)
            predictions = torch.argmax(act_probs, dim=1)
            total_correct += torch.sum((labels == predictions)).item()
    accuracy = total_correct / size
    image = data[0:1]
    _, locations, _, _ = modelRAM(image)
    draw_locations(image.cpu().numpy()[0][0], locations.detach().cpu().numpy()[0], epoch=epoch)
    return accuracy

if __name__ == "__main__":
    # training set : validation set : test set = 50000 : 10000 : 10000
    train_set               = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    trainnum                = len(train_set)
    indices                 = list(range(trainnum))
    valid_size              = 10000
    train_size              = trainnum - valid_size
    train_idx, valid_idx    = indices[valid_size:], indices[:valid_size]
    train_sampler           = SubsetRandomSampler(train_idx)
    valid_sampler           = SubsetRandomSampler(valid_idx)
    train_loader            = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
    valid_loader            = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=valid_sampler, **kwargs)
    test_loader             = torch.utils.data.DataLoader(datasets.MNIST('data', train=False, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True, **kwargs)

    modelRAM = ModelVT(location_size=args.location_size,
                       location_std=args.location_std,
                       num_classes=args.num_classes,
                       glimpse_size=args.glimpse_size,
                       num_glimpses=args.num_glimpses,
                       num_scales=args.num_scales,
                       feature_size=args.feature_size,
                       glimpse_feature_size=args.glimpse_feature_size,
                       hidden_size=args.hidden_size,
                       model_device=device).to(device)

    # Compute learning rate decay rate
    lr_decay_rate = args.lr / args.epochs
    # optimizer = optim.SGD(modelRAM.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(modelRAM.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, verbose=True, patience=5)

    celoss_fn = nn.CrossEntropyLoss()

    best_valid_accuracy, test_accuracy = 0, 0

    for epoch in range(1, args.epochs + 1):
        train(modelRAM, epoch, train_loader, celoss_fn)
        accuracy = test(modelRAM, epoch, valid_loader, valid_size)
        scheduler.step(accuracy)
        print('====> Validation set accuracy: {:.2%}'.format(accuracy))
        if accuracy > best_valid_accuracy:
            best_valid_accuracy = accuracy
            test_accuracy = test(modelRAM, epoch, test_loader, len(test_loader.dataset))
            # torch.save(modelRAM, 'save/best_model')
            print('====> Test set accuracy: {:.2%}'.format(test_accuracy))
        print('')

    print('====> Test set accuracy: {:.2%}'.format(test_accuracy))

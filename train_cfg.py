
"""
Training configuration.
"""

import argparse
import os

import torch
from torch import optim
from torchvision import transforms, datasets

from datasets.norb import smallNORB
from dynamic_capsules import DynamicCapsules
from em_capsules import em_standard, em_small, em_one
from loss import SpreadLoss, MarginLoss, CrossEntropyLoss
from spectral_capsules import spectral_capsules

# Available models
model_dict = {'dynamic': DynamicCapsules,
              'em-standard': em_standard,
              'em-small': em_small,
              'em-one': em_one,
              'spectral': spectral_capsules}

# Available loss functions
loss_functions = {'spread': SpreadLoss,
                  'margin': MarginLoss,
                  'crossent': CrossEntropyLoss}

optimizers = {'adam': optim.Adam,
              'sgd': optim.SGD}


# Argument parser
def arguments():
    parser = argparse.ArgumentParser(description='Capsule Networks')
    parser.add_argument('--model', '-m', metavar='M', default='dynamic',
                        choices=('em-standard', 'em-small', 'em-one',
                                 'spectral', 'dynamic', 'dense'),
                        help='Capsule model to train with'
                             ' (em-standard, em-small, em-one,'
                             ' spectral, dynamic, dense)')
    parser.add_argument('--dataset', default='mnist', metavar='D',
                        choices=('mnist', 'cifar10', 'smallNORB', 'svhn', 'fashion'),
                        help='The dataset to use for the experiment'
                             ' (mnist, cifar10, smallNORB, svhn, fashion)')
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N', help='mini-batch size (default: 100)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run for')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--routing', default=3, type=int, metavar='N',
                        help='number of routing iterations')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate (default: 0.1)')
    parser.add_argument('--optimizer', default='adam', metavar='O',
                        choices=('adam', 'sgd'),
                        help='The optimizer to use for training (adam, sgd)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-nesterov', action='store_true', default=False,
                        help='disable Nesterov momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--loss', default='spread', metavar='L',
                        choices=('spread', 'margin', 'crossent'),
                        help='loss to use (crossent, margin, spread)')
    parser.add_argument('--remake', action='store_true', default=False,
                        help='whether to include reconstruction loss')
    parser.add_argument('--remake-log-interval', default=None, type=int, metavar='N',
                        help='how many batches to wait before logging reconstructions')
    parser.add_argument('--log-interval', default=20, type=int, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-interval', default=1, metavar='N',
                        help='How often to save checkpoints (epochs).')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--disable-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', default=None, type=int, metavar='S',
                        help='seed')
    parser.add_argument('--manual-seed', default=None, type=int, metavar='S',
                        help='manual seed')

    parser.add_argument('--early-stopping-rounds', default=50, type=int, metavar='R',
                        help='number of rounds of no improvement before early stopping')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--root', default='./experiments', metavar='PATH',
                        help='root directory')
    parser.add_argument('--subdir', default='', metavar='PATH',
                        help='root subdirectory')
    parser.add_argument('--summary-dir', default='summaries', metavar='PATH',
                        help='where to save summaries')
    parser.add_argument('--dataset-dir', default='./datasets', metavar='PATH',
                        help='data containing folder')
    parser.add_argument('--save_data_dir', default='save_data',
                        help='folder to output images and model checkpoints')

    return parser


def get_dataloaders(args):
    path = os.path.join(args.dataset_dir, args.dataset)

    kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if not args.device == 'cuda' else {}

    # MNIST dataset
    if args.dataset == 'mnist':
        args.num_classes = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.RandomCrop(size=28, padding=2),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, transform=transforms.Compose([
                # transforms.RandomAffine(degrees=(90, -90)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    # CIFAR-10 dataset
    elif args.dataset == 'cifar10':
        args.num_classes = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(path, train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(path, train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    elif args.dataset == 'smallNORB':
        args.num_classes = 5
        train_loader = torch.utils.data.DataLoader(
            smallNORB(path, train=True, download=True,
                      transform=transforms.Compose([
                          # transforms.ColorJitter(brightness=32. / 255, contrast=0.8),  # Refer to Sara's code
                          transforms.Resize(48),
                          transforms.RandomCrop(32),
                          transforms.ToTensor(),
                          transforms.Normalize((0.7546,), (0.1755,))
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            smallNORB(path, train=False,
                      transform=transforms.Compose([
                          transforms.Resize(48),
                          transforms.CenterCrop(32),
                          transforms.ToTensor(),
                          transforms.Normalize((0.7546,), (0.1755,))
                      ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    elif args.dataset == 'svhn':
        args.num_classes = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=path, split='train', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]),
            ),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=path, split='test', download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]),
            ),
            batch_size=args.batch_size, shuffle=False, **kwargs)

    elif args.dataset == 'fashion':
        args.num_classes = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.RandomCrop(size=28, padding=2),
                                      transforms.ToTensor(),
                                      # transforms.Normalize((0.1307,), (0.3081,))
                                  ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)


    else:
        raise ValueError('Dataset not available.')

    args.input_shape = train_loader.dataset[0][0].shape

    return train_loader, test_loader


## Means, Std_dev
# MNIST/Fashion: (0.1307,), (0.3081,)
# smallNORB: (0.7546,), (0.1755,)
# ds = datasets.FashionMNIST("./datasets/mnist", train=True, download=True, transform=transforms.ToTensor())
# ds = smallNORB("./datasets/smallNORB", train=True, download=True, transform=transforms.ToTensor())


# print(ds.)
# ds.
# print(ds.mean() / 255)
# print(ds.std() / 255)
# print(train_set.train_data.std(axis=(0,1,2))/255)

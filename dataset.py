from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
import torch
import os
import pickle as pkl


def get_dataset(opts):
    save_dir = os.path.join('catch')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, '{}.pkl'.format(opts.dataset))
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            data = pkl.load(f)
            print('Load data from {}'.format(save_path))

            return data

    cifar10_norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    mnist_norm = ((0.1307,), (0.3081,))

    dat = {}
    if opts.dataset == "cifar10":

        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*cifar10_norm)])
        valid_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*cifar10_norm)])

        dat['train'] = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dat['test'] = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
    elif opts.dataset == 'mnist':
        train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mnist_norm)])
        valid_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mnist_norm)])
        dat['train'] = MNIST(root="./data", train=True, download=True, transform=train_transform)
        dat['test'] = MNIST(root="./data", train=False, download=True, transform=valid_transform)

    else:
        raise NotImplementedError
    data = {}
    for n in range(opts.num_task):
        data[n] = {}
        data[n]['train'] = {'x': [], 'y': []}
        data[n]['test'] = {'x': [], 'y': []}
    for s in ['train', 'test']:
        loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
        for image, target in loader:
            n = target.numpy()[0]
            nn = n // opts.class_per_task
            data[nn][s]['x'].append(image)
            data[nn][s]['y'].append(n % opts.class_per_task)
    for t in data.keys():
        for s in ['train', 'test']:
            data[t][s]['x'] = torch.stack(data[t][s]['x'], dim=0)
            data[t][s]['y'] = torch.LongTensor(data[t][s]['y']).view(-1)
    with open(save_path, 'wb') as f:
        pkl.dump(data, f)
        print('Save data into {}'.format(save_path))

    return data

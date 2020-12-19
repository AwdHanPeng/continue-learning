from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch


def get_dataset(opts):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    train_transform = transforms.Compose(normalize)
    valid_transform = transforms.Compose(normalize)

    dat = {}
    if opts.dataset == "cifar10":
        dat['train'] = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dat['test'] = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
    elif opts.dataset == 'mnist':
        dat['train'] = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dat['test'] = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)

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
    return data

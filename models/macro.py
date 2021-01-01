import torch.nn as nn
from torch.nn import functional as F
import torch
from copy import deepcopy


class CNNStack(nn.Module):
    def __init__(self, lis):
        '''

        :param lis: [(),()]  a list of tuple
        '''
        super().__init__()
        lis = deepcopy(lis)
        main_size = lis.pop(0)
        self.main_cnn = nn.Conv2d(*main_size)
        if len(lis) > 0:
            self.cnn_list = nn.ModuleList([torch.nn.Conv2d(*size) for size in lis])
        else:
            self.cnn_list = None
        self.relu = nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)

    def forward(self, data):
        data = data.squeeze(-4)
        x = self.main_cnn(data)
        if self.cnn_list:
            data_list = [adapt(x) for adapt in self.cnn_list]
            fuse_data = torch.mean(torch.stack(data_list, dim=0), dim=0)
            x = x + fuse_data
        output = self.maxpool(self.relu(x))
        return output


class MLPStack(nn.Module):
    def __init__(self, size):
        '''
        :param size: tuple()
        '''
        super().__init__()
        self.mlp = nn.Linear(*size)
        self.relu = nn.ReLU()

    def forward(self, data):
        if len(data.shape) == 4:
            data = data.view(data.size(0), -1)
        return self.relu(self.mlp(data))


class CNN(nn.Module):
    def __init__(self, configs, classify_in, opts):
        '''
        :param configs:
        config:[{'conv':[(),()]},'mlp':(),]
        dafault_config =
                for MLP:
                [{'mlp': (input_feature, self.args.mlp_size)}] + [{'mlp': (self.args.mlp_size, self.args.mlp_size)}] * (self.args.mlp_linear - 1)
                for CNN:
                [{'conv':[(input_size,64,4)]},{'conv':[(64,128,3)]},{'conv':[(128,256,2)]},{'mlp':((input_size-6)^2*256,2048)},{'mlp':(2048,2048)}]
        :param num_task:
        :param num_class:
        '''
        super().__init__()
        self.layers = nn.Sequential()
        for idx, config in enumerate(configs):
            sublayer = nn.Sequential()
            for key, size in config.items():
                if key == 'mlp':
                    sublayer.add_module(key, MLPStack(size))
                elif key == 'conv':
                    sublayer.add_module(key, CNNStack(size))
            self.layers.add_module('Stack{}'.format(idx), sublayer)
        self.classify = nn.ModuleList(
            [nn.Linear(classify_in, opts.num_class // opts.num_task) for _ in range(opts.num_task)])

    def forward(self, data):
        x = self.layers(data)
        output = [task(x) for task in self.classify]
        return output

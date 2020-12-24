import torch.nn as nn
from torch.nn import functional as F
import torch


class MLP(nn.Module):
    def __init__(self, configs, classify_in, opts):
        '''

        :param configs:
        config:[{'mlp':(512,512)},{'mlp':(512,512)}]
        
        :param num_task:
        :param num_class:
        '''
        super().__init__()
        self.layers = nn.Sequential()

        for idx, config in enumerate(configs):
            sublayer = nn.Sequential()
            for key, size in config.items():
                if key == 'mlp':
                    sublayer.add_module(key, nn.Linear(*size))
                elif key == 'conv':
                    raise NotImplemented
            self.layers.add_module('Stack{}'.format(idx), sublayer)

        self.classify = nn.ModuleList(
            [nn.Linear(classify_in, opts.num_class // opts.num_task) for _ in range(opts.num_task)])

    def forward(self, data):
        bs = data.shape[0]
        x = data.reshape(bs, -1)

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
        output = [task(x) for task in self.classify]
        return output

import torch.nn as nn
from torch.nn import functional as F
import torch


class StackedLSTMCell(nn.Module):
    def __init__(self, layers, size):
        super().__init__()
        self.lstm_num_layers = layers
        self.hidden_size = size
        self.input_size = size
        self.lstm_modules = nn.ModuleList([nn.LSTMCell(self.input_size, self.hidden_size, )
                                           for _ in range(self.lstm_num_layers)])

    def forward(self, inputs, hidden):
        prev_h, prev_c = hidden if hidden else (
            [torch.zeros(1, self.hidden_size).to(inputs.device)] * self.lstm_num_layers,
            [torch.zeros(1, self.hidden_size).to(inputs.device)] * self.lstm_num_layers)
        next_h, next_c = [], []
        inputs = inputs.view(1, -1)
        for i, m in enumerate(self.lstm_modules):
            curr_h, curr_c = m(inputs, (prev_h[i], prev_c[i]))
            next_c.append(curr_c)
            next_h.append(curr_h)
            # current implementation only supports batch size equals 1,
            # but the algorithm does not necessarily have this limitation
            inputs = curr_h[-1].view(1, -1)
        return next_h, next_c


class Controller(nn.Module):
    def __init__(self, args, task_num, ):
        '''
        选择空间 [new, reuse 0, adapt 0, reuse 1, adapt1 ,.....]
        :param args: 
        :param task_num: 
        :param adapt: 
        '''
        super().__init__()
        self.args = args
        self.task_scope = 1  # =>reuse
        self.general_scope = 1  # =>new
        if self.args.adapt: self.task_scope += 1
        if self.args.fuse: self.general_scope += 1

        self.choice_num = (task_num - 1) * self.task_scope + self.general_scope
        self.embedding = nn.Embedding(self.choice_num + 1, args.hidden)
        self.lstm = StackedLSTMCell(args.n_layers, args.hidden)
        self.choice = nn.Linear(args.hidden, self.choice_num)

    def create_mask(self, task):
        mask = [0] * self.choice_num
        mask[:self.general_scope + task * self.task_scope] = [1] * (self.general_scope + task * self.task_scope)
        assert len(mask) == self.choice_num
        return torch.tensor(mask)

    def forward(self, input, task, hidden=None):
        embed = self.embedding(input)
        hidden, cell = self.lstm(embed, hidden)
        mask = self.create_mask(task)
        logit = self.choice(hidden[-1].squeeze()).masked_fill_(mask=(mask.to(input.device) == 0), value=-1e9)
        probs = F.softmax(logit, dim=-1)
        # print(probs.tolist())
        # print(mask.tolist())
        return logit, (hidden, cell)

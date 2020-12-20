import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import datetime
from torch.nn import functional as F
from sklearn.metrics import accuracy_score
from copy import deepcopy


class Trainer:

    def __init__(self, model, task, args,
                 data):
        '''
        针对某个具体的子任务构建训练器，并根据传入的已训练好的参数进行初始化，最终返回该任务上达到的acc以及训练之后得到的模型参数
        :param model: 
        :param task: 
        :param args: 
        :param data: 
        '''
        self.eval_steps = args.eval_steps
        cuda_condition = torch.cuda.is_available() and args.with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        self.model = model.to(self.device)
        if args.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for model" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=args.cuda_devices)
        self.data = data
        self.params, self.params_name = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params.append(param)
                self.params_name.append(name)
        self.optim = Adam(self.params, lr=args.lr, )
        self.task = task
        self.shuffle = args.shuffle
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.loss_function = torch.nn.CrossEntropyLoss()

    def reload_checkpoint(self, dic):
        '''
        这里边传入要替换的参数，参数仅仅选择好，建立映射的过程在该函数内完成，可参照rat里的代码
        :param dict: 
        :return: 
        '''
        dic = deepcopy(dic)
        for key, value in self.model.state_dict().items():
            if key not in dic:
                dic[key] = value
        self.model.load_state_dict(dic)

    def history_eval(self, task_list):
        self.model.eval()
        acc_list = []
        for task in task_list:
            curren_test_data = self.data[task]['test']
            v_images = curren_test_data['x'].to(self.device)
            v_targets = curren_test_data['y'].to(self.device)
            logits = self.model(v_images)[task]
            predict = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
            v_targets, predict = v_targets.cpu(), predict.cpu()
            acc = accuracy_score(v_targets, predict)
            acc_list.append(acc)
        return acc_list

    def run(self):
        current_train_data, curren_test_data = self.data[self.task]['train'], self.data[self.task]['test']
        best_acc = 0
        model_dict = None
        for epoch in range(self.epochs):
            self.model.train()
            idx = np.arange(len(current_train_data['x']))
            if self.shuffle:
                np.random.shuffle(idx)
            idx = torch.LongTensor(idx)
            for i in range(0, len(idx), self.batch_size):
                if i + self.batch_size <= len(idx):
                    bs_idx = idx[i:i + self.batch_size]
                else:
                    bs_idx = idx[i:]
                images = current_train_data['x'][bs_idx].to(self.device)
                targets = current_train_data['y'][bs_idx].to(self.device)
                output = self.model(images)
                current_loss = self.loss_function(output[self.task], targets)
                self.optim.zero_grad()
                current_loss.backward()
                self.optim.step()

                if i // self.batch_size % self.eval_steps == 0:
                    self.model.eval()
                    v_images = curren_test_data['x'].to(self.device)
                    v_targets = curren_test_data['y'].to(self.device)
                    logits = self.model(v_images)[self.task]
                    predict = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
                    v_targets, predict = v_targets.cpu(), predict.cpu()
                    acc = accuracy_score(v_targets, predict)
                    if acc >= best_acc:
                        best_acc = acc
                        model_dict = self.model.state_dict()
        return best_acc, model_dict

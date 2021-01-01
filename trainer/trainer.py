import torch
import torch.nn as nn
from torch.optim import Adam, SGD
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
        self.data = data
        self.params_name = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params_name.append(name)
        self.task = task
        self.shuffle = args.shuffle
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.lr = args.lr
        self.args = args
        self.previous_dic = dict()
        self.lr_patience = args.lr_patience
        self.lr_factor = args.lr_factor

    def reload_checkpoint(self, dic):
        '''
        这里边传入要替换的参数，参数仅仅选择好，建立映射的过程在该函数内完成，可参照rat里的代码
        :param dict: 
        :return: 
        '''
        dic = deepcopy(dic)
        self.previous_dic = deepcopy(dic)
        if self.args.reuse_fixed:
            for key, value in dic.items():
                if 'Stack' in key:
                    self.params_name.remove(key)  # reuse之后不再学习
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

    def l2_loss(self):
        loss_reg = 0
        for key, value in self.previous_dic.items():
            if 'Stack' in key:
                loss_reg += torch.sum((value - self.model.state_dict(keep_vars=True)[key]).pow(2)) / 2
        return loss_reg

    def get_optim(self, params):
        if self.args.sgd:
            optim = SGD(params, lr=self.lr)
        else:
            optim = Adam(params, lr=self.lr)
        return optim

    def run(self, task_list=None):
        patience = self.lr_patience
        params, params_name = [], []
        for name, param in self.model.named_parameters():
            if name in self.params_name:
                params.append(param)
        optim = self.get_optim(params)
        current_train_data, curren_test_data = self.data[self.task]['train'], self.data[self.task]['test']
        best_acc, best_avg_acc = 0, 0
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
                if self.args.l2:
                    current_loss += self.l2_loss() * self.args.l2_weight
                optim.zero_grad()
                current_loss.backward()
                if self.args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                optim.step()

                if i // self.batch_size % self.eval_steps == 0:
                    self.model.eval()
                    v_images = curren_test_data['x'].to(self.device)
                    v_targets = curren_test_data['y'].to(self.device)
                    logits = self.model(v_images)[self.task]
                    predict = torch.argmax(F.softmax(logits, dim=-1), dim=-1)
                    v_targets, predict = v_targets.cpu(), predict.cpu()
                    acc = accuracy_score(v_targets, predict)

                    if self.args.back_eval and task_list:
                        # 综合考虑回测acc和当前acc
                        back_acc_list = self.history_eval(task_list)
                        back_acc_list.append(acc)
                        avg_acc = torch.mean(torch.tensor(back_acc_list))
                        if avg_acc >= best_avg_acc:
                            best_avg_acc = avg_acc
                            best_acc = acc
                            model_dict = self.model.state_dict()
                        else:
                            patience -= 1
                            if patience <= 0:
                                self.lr /= self.lr_factor
                                patience = self.lr_patience
                                optim = self.get_optim(params)
                    else:
                        if acc >= best_acc:
                            best_acc = acc
                            model_dict = self.model.state_dict()
                        else:
                            patience -= 1
                            if patience <= 0:
                                self.lr /= self.lr_factor
                                patience = self.lr_patience
                                optim = self.get_optim(params)
        return best_acc, model_dict

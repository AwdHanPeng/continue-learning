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
from .trainer import Trainer
from .models import MLP, Controller


class Mutator:

    def __init__(self, args, data, opts):
        '''
        需要完成几个任务，第一个是对初始任务，怎么考虑
        第二是从构建此表，最终完成采样过程
        第三是根据采样过程，构建出模型的size，选取出需要用到的参数
        :param model:
        :param task:
        :param args:
        :param data:
        '''
        self.args = args
        self.data = data
        self.opts = opts
        self.controller = Controller(args=self.args, task_num=self.opts.num_task, adapt=self.args.adapt)
        self.controller_optim = Adam(self.controller.parameters())
        self.tasks_config = []
        self.task_acc = []
        self.model_dict = []
        self.use_scope = 3 if self.args.adapt else 2

    def run(self):
        if self.args.base == 'mlp':
            self.run_mlp()
        else:
            raise NotImplemented

    def controller_sample(self, task):
        if self.args.base == 'mlp':
            steps = self.args.mlp_linear
        else:
            raise NotImplemented
        step_probs = []
        step_idx = []
        step_losses = []
        sample_idx = 0

        hidden = None
        for step in steps:
            logit, (hidden, cell) = self.controller(input=sample_idx, task=task, hidden=hidden)
            sample_idx = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            assert sample_idx < task * self.use_scope
            step_probs.append(probs)
            step_idx.append(sample_idx)
            step_losses.append(F.cross_entropy(logit, sample_idx))
        step_losses = torch.tensor(step_losses)
        return step_probs, step_idx, torch.mean(step_losses)

    def crop_model(self, step_idx, default_config):
        def get_layer_dict(dic, layer):
            '''
            从一个模型的参数中，取出某一个层的参数
            :param dic:
            :param layer:
            :return:
            '''
            # FIXME:
            pass

        cur_model_dict = []
        cur_model_config = []
        for layer, step in enumerate(range(step_idx)):
            # 选择空间 [new, reuse 0, adapt 0, reuse 1, adapt1 ,.....]

            if step == 0:
                choice = 0
                print('New {} layer para'.format(step))

                cur_model_config.append(default_config[layer])
            else:
                task_num = (step - 1) // (self.use_scope - 1)
                choice = (step - 1) % (self.use_scope - 1) + 1
                use_dict = self.model_dict[task_num]
                use_config = self.tasks_config[task_num]

                if choice == 1:
                    print('reuse {} layer para of task {}'.format(layer, task_num))
                    cur_model_dict.append(get_layer_dict(use_dict, layer))
                    cur_model_config.append(use_config[layer])
                elif self.use_scope == 3 and choice == 2:
                    print('Adapt {} layer para of task {}'.format(layer, task_num))
                    raise NotImplemented
        return cur_model_dict, cur_model_config

    def count_reward(self, cur_acc_lis, back_acc_list):
        '''
        
        :param cur_acc_lis: 当前任务上，不同采样过程中出现的acc
        :param back_acc_list: 当前采样的情况下，对历史任务的回测acc
        :return: 
        '''
        if len(cur_acc_lis) > 1:
            beta = (cur_acc_lis[-1] - cur_acc_lis[-2]) / cur_acc_lis[-2]
        else:
            beta = 0
        self.controller.eval()
        alpha = []
        assert len(history_acc_list) == len(self.task_acc)
        for origin_acc, eval_back_acc in zip(self.task_acc, back_acc_list):
            acc_drop = max(0, origin_acc - eval_back_acc)
            alpha.append(acc_drop / origin_acc)
        noise = 0.001
        alpha = torch.mean(torch.tensor(alpha)) + noise
        reward = 1 / alpha + beta
        return reward

    def run_mlp(self):
        if args.dataset == 'mnist':
            input_feature = 28 * 28
        elif args.dataset == 'cifar10':
            input_feature = 32 * 32
        else:
            input_feature = 0
            raise NotImplemented
        default_config = [{'mlp': (input_feature, self.args.mlp_size)}] + [
            {'mlp': (self.args.mlp_size, self.args.mlp_size)}] * (self.args.mlp_linear - 1)
        for task in self.opts.num_task:
            if task == 0:
                cur_model = MLP(default_config, self.args.mlp_size, self.opts)
                trainer = Trainer(model=cur_model, task=task, args=self.args, data=self.data)
                cur_acc, cur_model_dic = trainer.run()
                self.tasks_config.append(config)
                self.task_acc.append(cur_acc)
                self.model_dict.append(cur_model_dic)
            else:
                best_reward = 0
                cur_acc_lis = []
                cur_best_acc, cur_best_dic, cur_best_config = 0, None, None
                report_back_acc_list = None
                for _ in range(self.args.controller_steps):
                    self.controller.train()
                    step_probs, step_idx, sample_loss = self.controller_sample(task)
                    cur_model_dict, cur_model_config = self.crop_model(step_idx, default_config)
                    cur_model = MLP(cur_model_config, self.args.mlp_size, self.opts)
                    trainer = Trainer(model=cur_model, task=task, args=self.args, data=self.data)
                    Trainer.reload_checkpoint(cur_model_dict)
                    cur_acc, cur_model_dic = trainer.run()
                    cur_acc_lis.append(cur_acc)
                    back_acc_list = trainer.history_eval(task_list=[range(0, task)])
                    reward = self.count_reward(cur_acc_lis, back_acc_list)
                    if reward > best_reward:
                        # 通过判断当前reward的情况 来决定是否存模型 而不是仅根据当前采样出来的子模型的acc来决定
                        best_reward = reward
                        cur_best_dic = cur_model_dic
                        cur_best_acc = cur_acc
                        cur_best_config = cur_model_config
                        report_back_acc_list = back_acc_list

                    loss = sample_loss * reward
                    loss.backward()
                    self.controller_optim.step()
                print('After task {}, back eval acc is {}'.format(task, report_back_acc_list))
                self.tasks_config.append(cur_best_config)
                self.task_acc.append(cur_best_acc)
                self.model_dict.append(cur_best_dic)

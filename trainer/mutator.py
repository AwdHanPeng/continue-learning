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
from models import MLP, Controller
import datetime
import math
import itertools
import random


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
        self.controller_optim = Adam(self.controller.parameters(), lr=args.controller_lr)
        cuda_condition = torch.cuda.is_available() and args.with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        self.controller = self.controller.to(self.device)

        self.tasks_config = []
        self.task_acc = []
        self.model_dict = []
        self.use_scope = 2 if self.args.adapt else 1

        self.tensorboard_writer = SummaryWriter()
        self.iter = 0

    def run(self):
        if self.args.base == 'mlp':
            report_final_eval_acc, final_log, all_acc = self.run_mlp()
        else:
            raise NotImplemented
        print('Acc:')
        for items in report_final_eval_acc:
            s = ''
            for item in items:
                s += '%.3f\t' % item
            print(s)
        print(all_acc)
        print(final_log)
        print(self.args)

    def controller_sample(self, task):
        if self.args.base == 'mlp':
            steps = self.args.mlp_linear
        else:
            raise NotImplemented
        step_probs = []
        step_idx = []
        step_losses = []
        sample_idx = torch.tensor(0).view(-1).to(self.device)

        hidden = None
        for step in range(steps):
            logit, hidden = self.controller(input=sample_idx, task=task, hidden=hidden)
            if self.args.greedy > 0 and random.random() < self.args.greedy:
                sample_idx = torch.tensor(random.randint(0, task * self.use_scope)).to(self.device)
            else:
                sample_idx = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            assert sample_idx < task * self.use_scope + 1
            step_probs.append(F.softmax(logit, dim=-1).tolist())
            step_idx.append(sample_idx.item())
            step_losses.append(F.cross_entropy(logit.view(1, -1), sample_idx.view(-1)))
        step_losses = torch.stack(step_losses, dim=0)
        return step_probs, step_idx, torch.mean(step_losses)

    def crop_model(self, step_idx, default_config):
        def get_layer_dict(cur_model_dict, use_dict, layer):
            # 从一个模型的参数中，取出某一个层的参数
            for key, value in use_dict.items():
                if 'Stack{}'.format(layer) in key:
                    cur_model_dict[key] = value
            return cur_model_dict

        def init_dict(last_model_dict, cur_model_dict):
            # 将最近的一层的classify继承下来
            for key, value in last_model_dict.items():
                if 'classify' in key:
                    cur_model_dict[key] = value
            return cur_model_dict

        cur_model_dict = dict()
        cur_model_dict = init_dict(self.model_dict[-1], cur_model_dict)

        cur_model_config = []
        create_log = ''
        for layer, step in enumerate(step_idx):
            # 选择空间 [new, reuse 0, adapt 0, reuse 1, adapt1 ,.....]
            # step = step.item()
            if step == 0:
                choice = 0
                create_log += 'NEW      '.format(layer)
                cur_model_config.append(default_config[layer])
            else:
                task_num = (step - 1) // self.use_scope
                choice = (step - 1) % self.use_scope + 1
                use_dict = self.model_dict[task_num]
                use_config = self.tasks_config[task_num]
                if choice == 1:
                    create_log += 'REUSE from task {}        '.format(task_num)
                    cur_model_dict = get_layer_dict(cur_model_dict, use_dict, layer)
                    cur_model_config.append(use_config[layer])
                elif self.use_scope == 3 and choice == 2:
                    create_log += 'ADAPT from task {}        '.format(task_num)
                    raise NotImplemented
        assert len(cur_model_config) == len(step_idx)
        return cur_model_dict, cur_model_config, create_log

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
        alpha = []
        assert len(back_acc_list) == len(self.task_acc)
        # for origin_acc, eval_back_acc in zip(self.task_acc, back_acc_list):
        #     # acc_drop = max(0, origin_acc - eval_back_acc)
        #     acc_drop = origin_acc - eval_back_acc  # TODO, find better reward
        #     alpha.append(acc_drop / origin_acc)
        noise = 0.001
        # alpha = 1 / (torch.mean(torch.tensor(alpha)) + noise)
        # alpha = -1 * (torch.mean(torch.tensor(alpha))) #TODO, find better reward
        # alpha =  torch.sigmoid(-1 * (torch.mean(torch.tensor(alpha)))) - 0.5
        # alpha = -1 * (torch.mean(torch.tensor(alpha))) + 0.05
        # alpha = -1 * (torch.mean(torch.tensor(alpha))) + 0.5
        # alpha = -1 * (torch.max(torch.tensor(alpha))) + 0.1
        # reward = alpha + beta

        for origin_acc, eval_back_acc in zip(self.task_acc, back_acc_list):
            # acc_drop = max(0, origin_acc - eval_back_acc)
            acc_drop = eval_back_acc / origin_acc
            alpha.append(acc_drop)
        alpha = torch.mean(torch.tensor(alpha))
        # alpha = max(alpha - 0.8, 0.0) * 5.0  # 尝试设置了一个tolerance
        reward = alpha

        self.tensorboard_writer.add_scalar('Reward/Sum', reward, self.iter)
        self.tensorboard_writer.add_scalar('Reward/Alpha', alpha, self.iter)
        self.tensorboard_writer.add_scalar('Reward/Beta', beta, self.iter)
        self.iter += 1
        return reward.item()

    def run_mlp(self):
        final_log = ''
        report_final_eval_acc = [[0.0] * self.opts.num_task for _ in range(self.opts.num_task)]

        if self.args.dataset == 'mnist':
            input_feature = 28 * 28
        elif self.args.dataset == 'cifar10':
            input_feature = 32 * 32
        else:
            input_feature = 0
            raise NotImplemented
        default_config = [{'mlp': (input_feature, self.args.mlp_size)}] + [
            {'mlp': (self.args.mlp_size, self.args.mlp_size)}] * (self.args.mlp_linear - 1)
        for task in range(self.opts.num_task):
            print('--------------Create Config and Dict for task {}--------------'.format(task))
            if task == 0:
                cur_model = MLP(default_config, self.args.mlp_size, self.opts)
                trainer = Trainer(model=cur_model, task=task, args=self.args, data=self.data)
                cur_acc, cur_model_dic = trainer.run()
                self.tasks_config.append(default_config)
                self.task_acc.append(cur_acc)
                self.model_dict.append(cur_model_dic)
                print('Task{} Best Acc is {}'.format(task, cur_acc))
                report_final_eval_acc[task][:task + 1] = [cur_acc]
            else:
                best_reward = float('-inf')
                cur_acc_lis = []
                cur_best_acc, cur_best_dic, cur_best_config, best_create_log, step_probs = 0, None, None, None, None
                report_back_acc_list = None
                if self.args.upper_bound:
                    valid_idx = list(range(task + 1))
                    total_choice = list(itertools.product(valid_idx, repeat=self.args.mlp_linear)) * 5
                    total_step = len(total_choice)
                elif self.args.base_model:
                    total_choice = [[task] * self.args.mlp_linear]
                    total_step = 1
                else:
                    total_step = self.args.controller_steps
                for steps in range(total_step):
                    if self.args.upper_bound or self.args.base_model:
                        step_idx = list(total_choice[steps])
                    else:
                        self.controller.train()
                        step_probs, step_idx, sample_loss = self.controller_sample(task)
                    cur_model_dict, cur_model_config, create_log = self.crop_model(step_idx, default_config)
                    cur_model = MLP(cur_model_config, self.args.mlp_size, self.opts)
                    trainer = Trainer(model=cur_model, task=task, args=self.args, data=self.data)
                    trainer.reload_checkpoint(cur_model_dict)
                    cur_acc, cur_model_dic = trainer.run(task_list=list(range(0, task)))
                    cur_acc_lis.append(cur_acc)
                    back_acc_list = trainer.history_eval(task_list=list(range(0, task)))
                    reward = self.count_reward(cur_acc_lis, back_acc_list)
                    if steps % self.args.controller_logging_step == 0:
                        print('-------Logging at {} step for controller-------'.format(steps))
                        print(create_log)
                        print('Reward:{}. '.format(reward))
                        if step_probs:
                            for step_prob in step_probs:
                                print(step_prob)
                    if reward > best_reward:
                        best_reward = reward
                        cur_best_dic = cur_model_dic
                        cur_best_acc = cur_acc
                        cur_best_config = cur_model_config
                        report_back_acc_list = back_acc_list
                        best_create_log = create_log
                    if self.args.upper_bound or self.args.base_model:
                        pass
                    else:
                        loss = sample_loss * reward
                        loss.backward()
                        self.controller_optim.step()
                print(
                    '\033[95mAfter task {}'.format(task))
                print(best_create_log)
                final_log = final_log + best_create_log + '\n'
                print('best reward :{}\033[0m'.format(best_reward))
                self.tasks_config.append(cur_best_config)
                self.task_acc.append(cur_best_acc)
                self.model_dict.append(cur_best_dic)
                report_final_eval_acc[task][:len(report_back_acc_list) + 1] = report_back_acc_list + [cur_best_acc]
                if task == self.opts.num_task - 1:
                    all_acc = torch.mean(torch.tensor(report_back_acc_list + [cur_best_acc])).item()
        return report_final_eval_acc, final_log, all_acc

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset

import time
from utils import *
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Client(object):
    """Simulated federated learning client."""

    def __init__(self, client_id):
        self.client_id = client_id
        # 添加训练进度跟踪
        self.training_progress = {
            'fsmgda': {'loss_reductions': [], 'acc_improvements': []},
            'fedcmoo': {'loss_reductions': [], 'acc_improvements': []},
            'fsmgda_vr': {'loss_reductions': [], 'acc_improvements': []}
        }
    def __repr__(self):
        return 'Client #{}\n'.format(self.client_id)

        # Server interactions

    def download(self, argv):  # For possible future works
        # Download from the server.
        try:
            return argv.copy()
        except:
            return argv

    def upload(self, argv):  # For possible future works
        # Upload to the server
        try:
            return argv.copy()
        except:
            return argv

    def set_data(self, data, config):
        """Set the client's DataLoader with its own data points."""
        if config['experiment'] == 'QM9':
            from torch_geometric.loader import DataLoader as PyGDataLoader
            DataLoader = PyGDataLoader
        else:
            from torch.utils.data import DataLoader as TorchDataLoader
            DataLoader = TorchDataLoader
        batch_size = config['hyperparameters']['local_training']['batch_size']
        self.dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

    def configure(self, config):
        pass
        # To be implemented

    def test(self):
        # Perform local model testing - never used local testing
        raise NotImplementedError

    def get_optimizer(self, config, model):
        optim = config['hyperparameters']['local_training']['optimizer']
        lr = config['hyperparameters']['local_training']['local_lr']
        momentum = config['hyperparameters']['local_training']['local_momentum']
        model_params = []
        for task, m in model.items():
            model_params += list(m.parameters())

        if 'RMSprop' == optim:
            optimizer = torch.optim.RMSprop(model_params, lr=lr, momentum=momentum)
        elif 'Adam' == optim:
            optimizer = torch.optim.Adam(model_params, lr=lr)
        elif 'SGD' == optim:
            optimizer = torch.optim.SGD(model_params, lr=lr, momentum=momentum)
        return optimizer

    def evaluate_local_performance(self, global_model, experiment_module, tasks, config):
        """评估模型在本地数据上的性能"""
        model_device = config['model_device']
        boost_w_gpu = True if device == 'cuda' and model_device != 'cuda' else False
        return_device = 'cuda' if (model_device == 'cuda' or boost_w_gpu) else 'cpu'

        # 临时保存原始模型状态
        original_states = {}
        for m in global_model:
            original_states[m] = copy.deepcopy(global_model[m].state_dict())

        # 评估性能
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        loss_list = []
        acc_list = []
        loss_fn = experiment_module.get_loss()

        for m in global_model:
            global_model[m].eval()

        with torch.no_grad():
            batch = next(iter(self.dataloader))
            images = experiment_module.trainLoopPreprocess(batch[0].to(device))

            # 计算每个任务的损失和准确率
            batch_loss = 0.0
            batch_correct = 0
            batch_samples = 0

            for task in tasks:
                labels = batch[tasks.index(task) + 1].to(device)
                rep, _ = global_model['rep'](images, None)
                out, _ = global_model[task](rep, None)

                # 计算损失
                loss = loss_fn[task](out, labels)
                batch_loss += loss.item()
                loss_list.append(batch_loss)
                # 计算准确率
                _, predicted = torch.max(out.data, 1)
                batch_correct += (predicted == labels).sum().item()
                batch_samples += labels.size(0)
                batch_acc = batch_correct/batch_samples
                acc_list.append(batch_acc)

                total_loss += batch_loss / len(tasks)  # 平均任务损失
                total_correct += batch_acc / len(tasks)  # 平均任务准确率

        avg_loss = total_loss
        avg_accuracy = total_correct

        # 恢复原始模型状态
        for m in global_model:
            global_model[m].load_state_dict(original_states[m])
            global_model[m].train()

        return avg_loss, avg_accuracy, loss_list,acc_list

    def local_train(self, config, global_model, experiment_module, tasks, **kwargs):
        model_device = config['model_device']
        boost_w_gpu = True if device == 'cuda' and model_device != 'cuda' else False
        return_device = 'cuda' if (model_device == 'cuda' or boost_w_gpu) else 'cpu'

        # 记录训练开始时的性能
        algorithm = config['algorithm']
        if algorithm in ['fsmgda', 'fedcmoo', 'fsmgda_vr']:
            start_loss, start_acc,start_loss_list,start_acc_list = self.evaluate_local_performance(global_model, experiment_module, tasks, config)

        optimizer = self.get_optimizer(config, global_model)
        loss_fn = experiment_module.get_loss()

        if config['algorithm'] in ['fsmgda']:
            updates = {t: {'rep': None, t: None} for t in tasks}
            for temp in updates.keys():
                updates[temp]['rep'] = None
            batch_loss = 0.0
            batch_correct = 0
            for i,task in enumerate(tasks):
                optimizer = self.get_optimizer(config, global_model)
                initial_model = model_to_dict(global_model['rep'])
                initial_task_model = model_to_dict(global_model[task])

                local_update_counter = 0
                local_updates_finished_flag = False
                while not local_updates_finished_flag:
                    for batch in self.dataloader:
                        optimizer.zero_grad()
                        if local_update_counter == config['hyperparameters']['local_training']['nb_of_local_rounds']:
                            local_updates_finished_flag = True
                            break

                        images = experiment_module.trainLoopPreprocess(
                            batch[0].to(device))  # if device != config['data']['trainset_device'] else batch[0])
                        labels = batch[tasks.index(task) + 1].to(
                            device)  # if device != config['data']['trainset_device'] else batch[tasks.index(task) + 1]

                        rep, _ = global_model['rep'](images, None)
                        out, _ = global_model[task](rep, None)
                        loss = loss_fn[task](out, labels)
                        loss.backward()

                        # Normalize gradients if required
                        if config['algorithm_args'][config['algorithm']]['normalize_local_iters']:
                            total_norm = 0.0
                            for name, param in global_model['rep'].named_parameters():
                                if param.grad is not None:
                                    total_norm += param.grad.data.norm(2).item() ** 2
                            for name, param in global_model[task].named_parameters():
                                if param.grad is not None:
                                    total_norm += param.grad.data.norm(2).item() ** 2
                            total_norm = total_norm ** 0.5

                            # Normalize gradients
                            for name, param in global_model['rep'].named_parameters():
                                if param.grad is not None:
                                    param.grad.data.div_(total_norm)
                            for name, param in global_model[task].named_parameters():
                                if param.grad is not None:
                                    param.grad.data.div_(total_norm)

                        optimizer.step()
                        local_update_counter += 1

                with torch.no_grad():
                    final_model = model_to_dict(global_model['rep'])
                    final_task_model = model_to_dict(global_model[task])
                    [reset_gradients(m) for m in [global_model['rep'], global_model[task]]]


                    updates[task]['rep'] = {name: (final_model[name] - initial_model[name]).to(return_device) for name
                                            in final_model}
                    updates[task][task] = {name: (final_task_model[name] - initial_task_model[name]).to(return_device)
                                           for name in final_task_model}
                    end_loss, end_acc, end_loss_list, end_acc_list = self.evaluate_local_performance(
                        global_model, experiment_module, tasks, config)
                    batch_loss += end_loss_list[i]/len(tasks)
                    batch_correct += end_acc_list[i]/len(tasks)

                    # Reset global model to initial state before starting next task training
                    dict_to_model(global_model['rep'], initial_model)
                    dict_to_model(global_model[task], initial_task_model)

            end_loss = batch_loss
            end_acc = batch_correct
            # 计算损失减少和准确率提升
            loss_reduction = start_loss - end_loss
            acc_improvement = end_acc - start_acc

            # 存储训练进度
            self.training_progress[algorithm]['loss_reductions'].append(loss_reduction)
            self.training_progress[algorithm]['acc_improvements'].append(acc_improvement)

            function_return = updates

            # 返回训练进度数据
            if isinstance(function_return, dict):
                function_return['training_progress'] = {
                    'loss_reduction': loss_reduction,
                    'acc_improvement': acc_improvement,
                    'start_loss': start_loss,
                    'start_acc': start_acc,
                    'end_loss': end_loss,
                    'end_acc': end_acc
                }

        elif config['algorithm'] == 'fedcmoo':
            current_weight = kwargs['current_weight']
            if kwargs['first_local_round'] == True:
                self.initial_round_gradients = {task: {'rep': None, task: None} for task in tasks}
                G = []
                for task in tasks:
                    initial_model = model_to_dict(global_model['rep'])  # 初始化模型参数
                    initial_task_model = model_to_dict(global_model[task])
                    batch = next(iter(self.dataloader))
                    images = experiment_module.trainLoopPreprocess(
                        batch[0].to(device))  # if device != config['data']['trainset_device'] else batch[0])
                    labels = batch[tasks.index(task) + 1].to(
                        device)  # if device != config['data']['trainset_device'] else batch[tasks.index(task) + 1]
                    optimizer.zero_grad()
                    rep, _ = global_model['rep'](images, None)
                    out, _ = global_model[task](rep, None)
                    loss = loss_fn[task](out, labels)
                    loss.backward()

                    # Collect self.initial_round_gradients - here we ignore changes in any moving type normalization e.g. group norm
                    # To include them we would need to calculate difference between final and initial model using state_dict
                    self.grad_saving_device = {True: 'cuda', False: model_device}[boost_w_gpu and kwargs['save_to_gpu']]

                    # self.grad_saving_device is generally model_device. I added an extra gpu utilization if gpu is not big enough to keep all
                    # clients' grads but we want to use at its max memory so that we keep as many grad in gpu as possible for faster training
                    with torch.no_grad():
                        self.initial_round_gradients[task]['rep'] = {
                            name: param.grad.clone().to(self.grad_saving_device) for name, param in
                            global_model['rep'].named_parameters()}
                        self.initial_round_gradients[task][task] = {name: param.grad.clone().to(self.grad_saving_device)
                                                                    for name, param in
                                                                    global_model[task].named_parameters()}

                # Normalize initial_round_gradients if required
                if config['algorithm_args'][config['algorithm']]['normalize_updates']:
                    for task in tasks:
                        # Compute L2 norm
                        total_norm = 0.0
                        for grad in self.initial_round_gradients[task]['rep'].values():
                            total_norm += grad.norm(2).item() ** 2
                        if config['algorithm_args'][config['algorithm']]['count_decoders']:
                            for grad in self.initial_round_gradients[task][task].values():
                                total_norm += grad.norm(2).item() ** 2
                        total_norm = total_norm ** 0.5

                        # Normalize gradients
                        for name in self.initial_round_gradients[task]['rep']:
                            self.initial_round_gradients[task]['rep'][name].div_(total_norm)
                        for name in self.initial_round_gradients[task][task]:
                            self.initial_round_gradients[task][task][name].div_(total_norm)

                with torch.no_grad():  # This is faster if there is enough space on gpu
                    G_T_G = torch.zeros((len(tasks), len(tasks)), dtype=torch.float32).cpu()
                    G = []
                    for task in tasks:
                        # Form the big vector v_j for task j
                        v_j = []
                        for name in self.initial_round_gradients[task]['rep']:
                            v_j.append(self.initial_round_gradients[task]['rep'][name].view(-1).clone().cpu())
                        if config['algorithm_args'][config['algorithm']]['count_decoders']:
                            for t in tasks:
                                if t == task:
                                    for name in self.initial_round_gradients[task][task]:
                                        v_j.append(
                                            self.initial_round_gradients[task][task][name].view(-1).clone().cpu())
                                else:
                                    v_j.append(
                                        torch.zeros_like(list(global_model[t].parameters())[0]).view(-1).clone().cpu())
                        G.append(torch.cat(v_j).cpu())
                    G = torch.cat([temp.reshape(1, -1) for temp in G]).T.cpu()

                function_return = (G, None)
            else:
                initial_model = model_to_dict(global_model['rep'])
                initial_task_model = {task: model_to_dict(global_model[task]) for task in tasks}
                for m in global_model:
                    global_model[m].to(self.grad_saving_device)

                for task in tasks:
                    for name, param in global_model['rep'].named_parameters():
                        if param.grad is None:
                            param.grad = self.initial_round_gradients[task]['rep'][name] * current_weight[task]
                        else:
                            param.grad += self.initial_round_gradients[task]['rep'][name] * current_weight[task]
                    for name, param in global_model[task].named_parameters():
                        temp = current_weight[task] if config['algorithm_args'][config['algorithm']][
                            'scale_decoders'] else 1
                        if param.grad is None:
                            param.grad = self.initial_round_gradients[task][task][name] * temp
                        else:
                            param.grad += self.initial_round_gradients[task][task][name] * temp

                for m in global_model:
                    global_model[m].to(device)

                optimizer.step()

                # #### myDel(self.initial_round_gradients)
                if self.grad_saving_device == 'cuda':
                    myAttrDel(self, 'initial_round_gradients')

                [reset_gradients(m) for m in [global_model['rep']] + [global_model[task] for task in tasks]]
                # myDel()

                # Continue with remaining local rounds

                local_updates_finished_flag, local_update_counter = False, 0

                # optimized code (summing directly losses) for no normalization and decoder scaling
                if not config['algorithm_args'][config['algorithm']]['normalize_updates'] and \
                        config['algorithm_args'][config['algorithm']]['scale_decoders']:
                    while not local_updates_finished_flag:
                        for batch in self.dataloader:
                            weighted_loss = 0.0
                            optimizer.zero_grad()
                            if local_update_counter == config['hyperparameters']['local_training'][
                                'nb_of_local_rounds'] - 1:
                                local_updates_finished_flag = True
                                break

                            # Compute the total weighted loss by summing over all task losses
                            images = experiment_module.trainLoopPreprocess(
                                batch[0].to(device))  # if device != config['data']['trainset_device'] else batch[0])
                            rep, _ = global_model['rep'](images, None)
                            for task in tasks:
                                labels = batch[tasks.index(task) + 1].to(
                                    device)  # if device != config['data']['trainset_device'] else batch[tasks.index(task) + 1]
                                out, _ = global_model[task](rep, None)
                                loss = loss_fn[task](out, labels)
                                weighted_loss += current_weight[task] * loss  # step10
                            # Backpropagate the combined weighted loss
                            weighted_loss.backward()
                            # Apply gradients and optimize
                            optimizer.step()
                            local_update_counter += 1

                else:
                    while not local_updates_finished_flag:
                        for batch in self.dataloader:
                            weighted_loss = 0.0
                            optimizer.zero_grad()
                            if local_update_counter == config['hyperparameters']['local_training'][
                                'nb_of_local_rounds'] - 1:
                                local_updates_finished_flag = True
                                break

                            task_gradients = {task: {'rep': [], 'task': []} for task in tasks}
                            if device == 'cuda':
                                torch.cuda.empty_cache()

                            for task in tasks:
                                images = experiment_module.trainLoopPreprocess(batch[0].to(
                                    device))  # if device != config['data']['trainset_device'] else batch[0])
                                labels = batch[tasks.index(task) + 1].to(
                                    device)  # if device != config['data']['trainset_device'] else batch[tasks.index(task) + 1]
                                rep, _ = global_model['rep'](images, None)
                                out, _ = global_model[task](rep, None)
                                loss = loss_fn[task](out, labels)
                                weighted_loss += current_weight[task] * loss

                                # Zero gradients for this task
                                optimizer.zero_grad()

                                # Compute gradients for the task
                                loss.backward(retain_graph=True)

                                # Store gradients for 'rep' and task model using state_dict
                                for name, param in global_model['rep'].state_dict(keep_vars=True).items():
                                    if param.grad is not None:
                                        # task_gradients[task]['rep'][name] = param.grad.data.clone()
                                        task_gradients[task]['rep'].append(param.grad.data.clone())
                                for name, param in global_model[task].state_dict(keep_vars=True).items():
                                    if param.grad is not None:
                                        task_gradients[task]['task'].append(param.grad.data.clone())

                            # Reset gradients after saving them
                            optimizer.zero_grad()
                            [reset_gradients(global_model[t]) for t in global_model]

                            # Normalize gradients if required
                            if config['algorithm_args'][config['algorithm']]['normalize_updates']:
                                for task in tasks:
                                    # Compute L2 norm
                                    total_norm = 0.0
                                    for grad in task_gradients[task]['rep']:
                                        total_norm += grad.norm(2).item() ** 2
                                    if config['algorithm_args'][config['algorithm']]['count_decoders']:
                                        for grad in task_gradients[task]['task']:
                                            total_norm += grad.norm(2).item() ** 2
                                    total_norm = total_norm ** 0.5
                                    # Normalize gradients
                                    for grad in task_gradients[task]['rep']:
                                        grad.div_(total_norm)
                                    for grad in task_gradients[task]['task']:
                                        grad.div_(total_norm)

                            # Apply weighted gradients
                            for task in tasks:
                                for param, grad in zip(global_model['rep'].parameters(), task_gradients[task]['rep']):
                                    if param.grad is None:
                                        param.grad = grad * current_weight[task]
                                    else:
                                        param.grad += grad * current_weight[task]
                                temp = current_weight[task] if config['algorithm_args'][config['algorithm']][
                                    'scale_decoders'] else 1
                                for param, grad in zip(global_model[task].parameters(), task_gradients[task]['task']):
                                    if param.grad is None:
                                        param.grad = grad * temp
                                    else:
                                        param.grad += grad * temp
                            optimizer.step()
                            local_update_counter += 1

                with torch.no_grad():
                    final_model = model_to_dict(global_model['rep'])
                    final_task_model = {task: model_to_dict(global_model[task]) for task in tasks}
                    [reset_gradients(global_model[t]) for t in global_model]

                function_return = {'rep': {
                    name: (final_model[name] - initial_model[name]).to({True: device, False: model_device}[boost_w_gpu])
                    for name in final_model}, **{task: {
                    name: (final_task_model[task][name] - initial_task_model[task][name]).to(
                        {True: device, False: model_device}[boost_w_gpu]) for name in final_task_model[task]} for task
                    in tasks}}

        elif config['algorithm'] == 'fedcmoo_pref':
            current_weight = kwargs['current_weight']
            if kwargs['first_local_round'] == True:
                self.initial_round_gradients = {task: {'rep': None, task: None} for task in tasks}
                G = []
                for task in tasks:
                    initial_model = model_to_dict(global_model['rep'])
                    initial_task_model = model_to_dict(global_model[task])
                    batch = next(iter(self.dataloader))
                    images = experiment_module.trainLoopPreprocess(
                        batch[0].to(device))  # if device != config['data']['trainset_device'] else batch[0])
                    labels = batch[tasks.index(task) + 1].to(
                        device)  # if device != config['data']['trainset_device'] else batch[tasks.index(task) + 1]
                    optimizer.zero_grad()
                    rep, _ = global_model['rep'](images, None)
                    out, _ = global_model[task](rep, None)
                    loss = loss_fn[task](out, labels)
                    loss.backward()
                    # Collect self.initial_round_gradients - here we ignore changes in any moving type normalization e.g. group norm
                    # To include them we would need to calculate difference between final and initial model using state_dict
                    self.grad_saving_device = {True: 'cuda', False: model_device}[boost_w_gpu and kwargs['save_to_gpu']]

                    # self.grad_saving_device is generally model_device. I added an extra gpu utilization if gpu is not big enough to keep all
                    # clients' grads but we want to use at its max memory so that we keep as many grad in gpu as possible for faster training
                    with torch.no_grad():
                        self.initial_round_gradients[task]['rep'] = {
                            name: param.grad.clone().to(self.grad_saving_device) for name, param in
                            global_model['rep'].named_parameters()}
                        self.initial_round_gradients[task][task] = {name: param.grad.clone().to(self.grad_saving_device)
                                                                    for name, param in
                                                                    global_model[task].named_parameters()}
                # Normalize initial_round_gradients if required
                if config['algorithm_args'][config['algorithm']]['normalize_updates']:
                    for task in tasks:
                        # Compute L2 norm
                        total_norm = 0.0
                        for grad in self.initial_round_gradients[task]['rep'].values():
                            total_norm += grad.norm(2).item() ** 2
                        if config['algorithm_args'][config['algorithm']]['count_decoders']:
                            for grad in self.initial_round_gradients[task][task].values():
                                total_norm += grad.norm(2).item() ** 2
                        total_norm = total_norm ** 0.5

                        # Normalize gradients
                        for name in self.initial_round_gradients[task]['rep']:
                            self.initial_round_gradients[task]['rep'][name].div_(total_norm)
                        for name in self.initial_round_gradients[task][task]:
                            self.initial_round_gradients[task][task][name].div_(total_norm)
                with torch.no_grad():  # This is faster if there is enough space on gpu
                    G_T_G = torch.zeros((len(tasks), len(tasks)), dtype=torch.float32).cpu()
                    G = []
                    for task in tasks:
                        # Form the big vector v_j for task j
                        v_j = []
                        for name in self.initial_round_gradients[task]['rep']:
                            v_j.append(self.initial_round_gradients[task]['rep'][name].view(-1).clone().cpu())
                        if config['algorithm_args'][config['algorithm']]['count_decoders']:
                            for t in tasks:
                                if t == task:
                                    for name in self.initial_round_gradients[task][task]:
                                        v_j.append(
                                            self.initial_round_gradients[task][task][name].view(-1).clone().cpu())
                                else:
                                    v_j.append(
                                        torch.zeros_like(list(global_model[t].parameters())[0]).view(-1).clone().cpu())
                        G.append(torch.cat(v_j).cpu())
                    G = torch.cat([temp.reshape(1, -1) for temp in G]).T.cpu()

                function_return = (G, None)
            else:
                initial_model = model_to_dict(global_model['rep'])
                initial_task_model = {task: model_to_dict(global_model[task]) for task in tasks}
                for m in global_model:
                    global_model[m].to(self.grad_saving_device)
                    #
                for task in tasks:
                    for name, param in global_model['rep'].named_parameters():
                        if param.grad is None:
                            param.grad = self.initial_round_gradients[task]['rep'][name] * current_weight[task]
                        else:
                            param.grad += self.initial_round_gradients[task]['rep'][name] * current_weight[task]
                    for name, param in global_model[task].named_parameters():
                        temp = current_weight[task] if config['algorithm_args'][config['algorithm']][
                            'scale_decoders'] else 1
                        if param.grad is None:
                            param.grad = self.initial_round_gradients[task][task][name] * temp
                        else:
                            param.grad += self.initial_round_gradients[task][task][name] * temp

                for m in global_model:
                    global_model[m].to(device)

                optimizer.step()

                if self.grad_saving_device == 'cuda':
                    myAttrDel(self, 'initial_round_gradients')

                [reset_gradients(m) for m in [global_model['rep']] + [global_model[task] for task in tasks]]

                # Continue with remaining local rounds

                local_updates_finished_flag, local_update_counter = False, 0

                # optimized code (summing directly losses) for no normalization and decoder scaling
                if not config['algorithm_args'][config['algorithm']]['normalize_updates'] and \
                        config['algorithm_args'][config['algorithm']]['scale_decoders']:
                    while not local_updates_finished_flag:
                        for batch in self.dataloader:
                            weighted_loss = 0.0
                            optimizer.zero_grad()
                            if local_update_counter == config['hyperparameters']['local_training'][
                                'nb_of_local_rounds'] - 1:
                                local_updates_finished_flag = True
                                break

                            # Compute the total weighted loss by summing over all task losses
                            images = experiment_module.trainLoopPreprocess(
                                batch[0].to(device))  # if device != config['data']['trainset_device'] else batch[0])
                            rep, _ = global_model['rep'](images, None)
                            for task in tasks:
                                labels = batch[tasks.index(task) + 1].to(
                                    device)  # if device != config['data']['trainset_device'] else batch[tasks.index(task) + 1]
                                out, _ = global_model[task](rep, None)
                                loss = loss_fn[task](out, labels)
                                weighted_loss += current_weight[task] * loss
                            # Backpropagate the combined weighted loss
                            weighted_loss.backward()
                            # Apply gradients and optimize
                            optimizer.step()
                            local_update_counter += 1
                else:
                    while not local_updates_finished_flag:
                        for batch in self.dataloader:
                            weighted_loss = 0.0
                            optimizer.zero_grad()
                            if local_update_counter == config['hyperparameters']['local_training'][
                                'nb_of_local_rounds'] - 1:
                                local_updates_finished_flag = True
                                break

                            task_gradients = {task: {'rep': [], 'task': []} for task in tasks}
                            if device == 'cuda':
                                torch.cuda.empty_cache()

                            for task in tasks:
                                images = experiment_module.trainLoopPreprocess(batch[0].to(
                                    device))  # if device != config['data']['trainset_device'] else batch[0])
                                labels = batch[tasks.index(task) + 1].to(
                                    device)  # if device != config['data']['trainset_device'] else batch[tasks.index(task) + 1]
                                rep, _ = global_model['rep'](images, None)
                                out, _ = global_model[task](rep, None)
                                loss = loss_fn[task](out, labels)
                                weighted_loss += current_weight[task] * loss

                                # Zero gradients for this task
                                optimizer.zero_grad()

                                # Compute gradients for the task
                                loss.backward(retain_graph=True)

                                # Store gradients for 'rep' and task model using state_dict
                                for name, param in global_model['rep'].state_dict(keep_vars=True).items():
                                    if param.grad is not None:
                                        # task_gradients[task]['rep'][name] = param.grad.data.clone()
                                        task_gradients[task]['rep'].append(param.grad.data.clone())
                                for name, param in global_model[task].state_dict(keep_vars=True).items():
                                    if param.grad is not None:
                                        task_gradients[task]['task'].append(param.grad.data.clone())

                            # Reset gradients after saving them
                            optimizer.zero_grad()
                            [reset_gradients(global_model[t]) for t in global_model]

                            # Normalize gradients if required
                            if config['algorithm_args'][config['algorithm']]['normalize_updates']:
                                for task in tasks:
                                    # Compute L2 norm
                                    total_norm = 0.0
                                    for grad in task_gradients[task]['rep']:
                                        total_norm += grad.norm(2).item() ** 2
                                    if config['algorithm_args'][config['algorithm']]['count_decoders']:
                                        for grad in task_gradients[task]['task']:
                                            total_norm += grad.norm(2).item() ** 2
                                    total_norm = total_norm ** 0.5
                                    # Normalize gradients
                                    for grad in task_gradients[task]['rep']:
                                        grad.div_(total_norm)
                                    for grad in task_gradients[task]['task']:
                                        grad.div_(total_norm)

                            # Apply weighted gradients
                            for task in tasks:
                                for param, grad in zip(global_model['rep'].parameters(), task_gradients[task]['rep']):
                                    if param.grad is None:
                                        param.grad = grad * current_weight[task]
                                    else:
                                        param.grad += grad * current_weight[task]
                                temp = current_weight[task] if config['algorithm_args'][config['algorithm']][
                                    'scale_decoders'] else 1
                                for param, grad in zip(global_model[task].parameters(), task_gradients[task]['task']):
                                    if param.grad is None:
                                        param.grad = grad * temp
                                    else:
                                        param.grad += grad * temp
                            optimizer.step()
                            local_update_counter += 1

                with torch.no_grad():
                    final_model = model_to_dict(global_model['rep'])
                    final_task_model = {task: model_to_dict(global_model[task]) for task in tasks}
                    [reset_gradients(global_model[t]) for t in global_model]

                function_return = {'rep': {
                    name: (final_model[name] - initial_model[name]).to({True: device, False: model_device}[boost_w_gpu])
                    for name in final_model}, **{task: {
                    name: (final_task_model[task][name] - initial_task_model[task][name]).to(
                        {True: device, False: model_device}[boost_w_gpu]) for name in final_task_model[task]} for task
                    in tasks}}

        elif config['algorithm'] == 'fedadam':
            current_weight = kwargs['current_weight']
            if kwargs['first_local_round'] == True:
                self.initial_round_gradients = {task: {'rep': None, task: None} for task in tasks}
                G = []
                for task in tasks:
                    initial_model = model_to_dict(global_model['rep'])
                    initial_task_model = model_to_dict(global_model[task])
                    batch = next(iter(self.dataloader))
                    images = experiment_module.trainLoopPreprocess(
                        batch[0].to(device))  # if device != config['data']['trainset_device'] else batch[0])
                    labels = batch[tasks.index(task) + 1].to(
                        device)  # if device != config['data']['trainset_device'] else batch[tasks.index(task) + 1]
                    optimizer.zero_grad()
                    rep, _ = global_model['rep'](images, None)
                    out, _ = global_model[task](rep, None)
                    loss = loss_fn[task](out, labels)
                    loss.backward()

                    # Collect self.initial_round_gradients - here we ignore changes in any moving type normalization e.g. group norm
                    # To include them we would need to calculate difference between final and initial model using state_dict
                    self.grad_saving_device = {True: 'cuda', False: model_device}[boost_w_gpu and kwargs['save_to_gpu']]

                    # self.grad_saving_device is generally model_device. I added an extra gpu utilization if gpu is not big enough to keep all
                    # clients' grads but we want to use at its max memory so that we keep as many grad in gpu as possible for faster training
                    with torch.no_grad():
                        self.initial_round_gradients[task]['rep'] = {
                            name: param.grad.clone().to(self.grad_saving_device) for name, param in
                            global_model['rep'].named_parameters()}
                        self.initial_round_gradients[task][task] = {name: param.grad.clone().to(self.grad_saving_device)
                                                                    for name, param in
                                                                    global_model[task].named_parameters()}

                # Normalize initial_round_gradients if required
                if config['algorithm_args'][config['algorithm']]['normalize_updates']:
                    for task in tasks:
                        # Compute L2 norm
                        total_norm = 0.0
                        for grad in self.initial_round_gradients[task]['rep'].values():
                            total_norm += grad.norm(2).item() ** 2
                        if config['algorithm_args'][config['algorithm']]['count_decoders']:
                            for grad in self.initial_round_gradients[task][task].values():
                                total_norm += grad.norm(2).item() ** 2
                        total_norm = total_norm ** 0.5

                        # Normalize gradients
                        for name in self.initial_round_gradients[task]['rep']:
                            self.initial_round_gradients[task]['rep'][name].div_(total_norm)
                        for name in self.initial_round_gradients[task][task]:
                            self.initial_round_gradients[task][task][name].div_(total_norm)

                with torch.no_grad():  # This is faster if there is enough space on gpu
                    G_T_G = torch.zeros((len(tasks), len(tasks)), dtype=torch.float32).cpu()
                    G = []
                    for task in tasks:
                        # Form the big vector v_j for task j
                        v_j = []
                        for name in self.initial_round_gradients[task]['rep']:
                            v_j.append(self.initial_round_gradients[task]['rep'][name].view(-1).clone().cpu())
                        if config['algorithm_args'][config['algorithm']]['count_decoders']:
                            for t in tasks:
                                if t == task:
                                    for name in self.initial_round_gradients[task][task]:
                                        v_j.append(
                                            self.initial_round_gradients[task][task][name].view(-1).clone().cpu())
                                else:
                                    v_j.append(
                                        torch.zeros_like(list(global_model[t].parameters())[0]).view(-1).clone().cpu())
                        G.append(torch.cat(v_j).cpu())
                    G = torch.cat([temp.reshape(1, -1) for temp in G]).T.cpu()

                    # 初始化
                function_return = (G, None)

            elif kwargs['initial_c_local'] == True:
                initial_model = model_to_dict(global_model['rep'])
                local_updates_finished_flag, local_update_counter = False, 0
                while not local_updates_finished_flag:
                    for batch in self.dataloader:
                        weighted_loss = 0.0
                        optimizer.zero_grad()
                        if local_update_counter == config['hyperparameters']['local_training'][
                            'nb_of_local_rounds'] - 1:
                            local_updates_finished_flag = True
                            break

                        if device == 'cuda':
                            torch.cuda.empty_cache()

                        for task in tasks:
                            images = experiment_module.trainLoopPreprocess(batch[0].to(
                                device))  # if device != config['data']['trainset_device'] else batch[0])
                            labels = batch[tasks.index(task) + 1].to(
                                device)  # if device != config['data']['trainset_device'] else batch[tasks.index(task) + 1]
                            rep, _ = global_model['rep'](images, None)
                            out, _ = global_model[task](rep, None)
                            loss = loss_fn[task](out, labels)
                            weighted_loss += current_weight[task] * loss

                            # Zero gradients for this task
                            optimizer.zero_grad()

                            # Compute gradients for the task
                            loss.backward(retain_graph=True)

                            # Store gradients for 'rep' and task model using state_dict
                            for name, param in global_model['rep'].state_dict(keep_vars=True).items():
                                if initial_model[name].grad is None:
                                    initial_model[name].grad = param.grad.data.clone() * \
                                                               current_weight[task]
                                else:
                                    initial_model[name].grad += param.grad.data.clone() * \
                                                                current_weight[task]

                        local_update_counter += 1

                function_return = [
                    (initial_model[name].grad / config['hyperparameters']['local_training'][
                        'nb_of_local_rounds']).to({True: device, False: model_device}[boost_w_gpu]) for name in
                    initial_model]

            else:
                initial_model = model_to_dict(global_model['rep'])
                update_c_model = model_to_dict(global_model['rep'])
                initial_task_model = {task: model_to_dict(global_model[task]) for task in tasks}
                for m in global_model:
                    global_model[m].to(self.grad_saving_device)

                for task in tasks:
                    for name, param in global_model['rep'].named_parameters():
                        if param.grad is None:
                            param.grad = self.initial_round_gradients[task]['rep'][name] * current_weight[task]
                            update_c_model[name].grad = self.initial_round_gradients[task]['rep'][name] * \
                                                        current_weight[task]
                        else:
                            param.grad += self.initial_round_gradients[task]['rep'][name] * current_weight[task]
                            update_c_model[name].grad += self.initial_round_gradients[task]['rep'][name] * \
                                                         current_weight[task]
                    for name, param in global_model[task].named_parameters():
                        temp = current_weight[task] if config['algorithm_args'][config['algorithm']][
                            'scale_decoders'] else 1
                        if param.grad is None:
                            param.grad = self.initial_round_gradients[task][task][name] * temp
                        else:
                            param.grad += self.initial_round_gradients[task][task][name] * temp

                for m in global_model:
                    global_model[m].to(device)

                optimizer.step()

                # #### myDel(self.initial_round_gradients)
                if self.grad_saving_device == 'cuda':
                    myAttrDel(self, 'initial_round_gradients')

                [reset_gradients(m) for m in [global_model['rep']] + [global_model[task] for task in tasks]]
                # myDel()

                # Continue with remaining local rounds

                local_updates_finished_flag, local_update_counter = False, 0

                # optimized code (summing directly losses) for no normalization and decoder scaling
                if not config['algorithm_args'][config['algorithm']]['normalize_updates'] and \
                        config['algorithm_args'][config['algorithm']]['scale_decoders']:
                    while not local_updates_finished_flag:
                        for batch in self.dataloader:
                            weighted_loss = 0.0
                            optimizer.zero_grad()
                            if local_update_counter == config['hyperparameters']['local_training'][
                                'nb_of_local_rounds'] - 1:
                                local_updates_finished_flag = True
                                break

                            # Compute the total weighted loss by summing over all task losses
                            images = experiment_module.trainLoopPreprocess(
                                batch[0].to(device))  # if device != config['data']['trainset_device'] else batch[0])
                            rep, _ = global_model['rep'](images, None)
                            for task in tasks:
                                labels = batch[tasks.index(task) + 1].to(
                                    device)  # if device != config['data']['trainset_device'] else batch[tasks.index(task) + 1]
                                out, _ = global_model[task](rep, None)
                                loss = loss_fn[task](out, labels)
                                weighted_loss += current_weight[task] * loss  # step10
                            # Backpropagate the combined weighted loss
                            weighted_loss.backward()
                            # Apply gradients and optimize
                            optimizer.step()
                            local_update_counter += 1

                else:
                    while not local_updates_finished_flag:
                        for batch in self.dataloader:
                            weighted_loss = 0.0
                            optimizer.zero_grad()
                            if local_update_counter == config['hyperparameters']['local_training'][
                                'nb_of_local_rounds'] - 1:
                                local_updates_finished_flag = True
                                break

                            task_gradients = {task: {'rep': [], 'task': []} for task in tasks}
                            if device == 'cuda':
                                torch.cuda.empty_cache()

                            for task in tasks:
                                images = experiment_module.trainLoopPreprocess(batch[0].to(
                                    device))  # if device != config['data']['trainset_device'] else batch[0])
                                labels = batch[tasks.index(task) + 1].to(
                                    device)  # if device != config['data']['trainset_device'] else batch[tasks.index(task) + 1]
                                rep, _ = global_model['rep'](images, None)
                                out, _ = global_model[task](rep, None)
                                loss = loss_fn[task](out, labels)
                                weighted_loss += current_weight[task] * loss

                                # Zero gradients for this task
                                optimizer.zero_grad()

                                # Compute gradients for the task
                                loss.backward(retain_graph=True)

                                # Store gradients for 'rep' and task model using state_dict
                                for name, param in global_model['rep'].state_dict(keep_vars=True).items():
                                    if param.grad is not None:
                                        # task_gradients[task]['rep'][name] = param.grad.data.clone()
                                        task_gradients[task]['rep'].append(param.grad.data.clone())
                                        update_c_model[name].grad += param.grad.data.clone() * \
                                                                     current_weight[task]
                                for name, param in global_model[task].state_dict(keep_vars=True).items():
                                    if param.grad is not None:
                                        task_gradients[task]['task'].append(param.grad.data.clone())

                            # Reset gradients after saving them
                            optimizer.zero_grad()
                            [reset_gradients(global_model[t]) for t in global_model]

                            # Apply weighted gradients
                            avg_task = 1 / len(tasks)
                            beta = config['algorithm_args'][config['algorithm']]['beta']
                            control_momentum = config['algorithm_args'][config['algorithm']]['control_momentum']
                            for task in tasks:
                                for param, grad, c_global, g_global, c_local in zip(global_model['rep'].parameters(),
                                                                                    task_gradients[task]['rep'],
                                                                                    kwargs['c_global'],
                                                                                    kwargs['g_global'],
                                                                                    kwargs['c_local']):
                                    # if param.grad is None:
                                    grad.data = beta * (grad * current_weight[task]) + avg_task * (
                                            beta * control_momentum * (c_global - c_local) + (1 - beta) * g_global)
                                    # else:
                                    #     param.grad += beta * (grad * current_weight[task])+avg_task * (beta *control_momentum* (c_global-c_local)+(1-beta) * g_global)

                            # Normalize gradients if required
                            if config['algorithm_args'][config['algorithm']]['normalize_grad']:
                                total_norm = 0.0
                                rep_gradients_norm = [torch.zeros_like(param).to(device) for param in
                                                      global_model['rep'].parameters()]
                                task_gradients_norm = [torch.zeros_like(param).to(device) for param in
                                                       global_model['rep'].parameters()]
                                for task in tasks:
                                    for param, grad in zip(rep_gradients_norm, task_gradients[task]['rep']):
                                        param += grad * current_weight[task]

                                for grad in rep_gradients_norm:
                                    total_norm += grad.norm(2).item() ** 2
                                total_norm = total_norm ** 0.5 + 1e-8
                                # 是否加任务端的模

                            for task in tasks:
                                for param, grad in zip(global_model['rep'].parameters(), task_gradients[task]['rep']):
                                    if param.grad is None:
                                        param.grad = grad / total_norm
                                    else:
                                        param.grad += grad / total_norm
                                temp = current_weight[task] if config['algorithm_args'][config['algorithm']][
                                    'scale_decoders'] else 1
                                for param, grad in zip(global_model[task].parameters(), task_gradients[task]['task']):
                                    if param.grad is None:
                                        param.grad = grad * temp
                                    else:
                                        param.grad += grad * temp

                            optimizer.step()
                            local_update_counter += 1

                with torch.no_grad():
                    final_model = model_to_dict(global_model['rep'])
                    final_task_model = {task: model_to_dict(global_model[task]) for task in tasks}
                    [reset_gradients(global_model[t]) for t in global_model]

                    # 更新本地变量c
                    # y_delta = [(final_model[name] - initial_model[name]).to({True: device, False: model_device}[boost_w_gpu]) for name in final_model]
                    coef = 1 / config['hyperparameters']['local_training']['nb_of_local_rounds']
                    # for c_l, diff, c_g, g_global in zip(kwargs['c_local'], y_delta, kwargs['c_global'], kwargs['g_global']):
                    #     c_plus.append(c_l - c_g - coef * (g_global * (1-config['algorithm_args'][config['algorithm']]['beta']) + diff))
                    c_plus = [
                        (update_c_model[name].grad.data * coef).to({True: device, False: model_device}[boost_w_gpu]) for
                        name in update_c_model]

                    c_local_update = c_plus

                    # 更新c_delta，即c^t_i − c^{t−1}_i
                    c_delta = []
                    for c_p, c_l in zip(c_plus, kwargs['c_local']):
                        c_delta.append(c_p - c_l)

                function_return = {'rep': {
                    name: (final_model[name] - initial_model[name]).to({True: device, False: model_device}[boost_w_gpu])
                    for name in final_model}, **{task: {
                    name: (final_task_model[task][name] - initial_task_model[task][name]).to(
                        {True: device, False: model_device}[boost_w_gpu]) for name in final_task_model[task]} for task
                    in tasks}, 'c_local': c_local_update, 'g_global': g_global, 'c_delta': c_delta}

        if config['algorithm'] in ['fsmgda_vr']:
            if kwargs['initial_d'] == True:
                self.updates = {t: {'rep': {name: None for name in model_to_dict(global_model['rep'])},
                               t: {name: None for name in model_to_dict(global_model[t])}} for t in tasks}

                for task in tasks:
                    initial_model = model_to_dict(global_model['rep'])
                    initial_task_model = model_to_dict(global_model[task])

                    local_update_counter = 0
                    local_updates_finished_flag = False
                    while not local_updates_finished_flag:
                        for batch in self.dataloader:
                            if local_update_counter == config['hyperparameters']['local_training'][
                                'nb_of_local_rounds']:
                                local_updates_finished_flag = True
                                break

                            images = experiment_module.trainLoopPreprocess(
                                batch[0].to(device))  # if device != config['data']['trainset_device'] else batch[0])
                            labels = batch[tasks.index(task) + 1].to(
                                device)  # if device != config['data']['trainset_device'] else batch[tasks.index(task) + 1]

                            rep, _ = global_model['rep'](images, None)
                            out, _ = global_model[task](rep, None)
                            loss = loss_fn[task](out, labels)
                            loss.backward()

                            # Normalize gradients if required
                            if config['algorithm_args'][config['algorithm']]['normalize_local_iters']:
                                total_norm = 0.0
                                for name, param in global_model['rep'].named_parameters():
                                    if param.grad is not None:
                                        total_norm += param.grad.data.norm(2).item() ** 2
                                for name, param in global_model[task].named_parameters():
                                    if param.grad is not None:
                                        total_norm += param.grad.data.norm(2).item() ** 2
                                total_norm = total_norm ** 0.5

                                # Normalize gradients
                                for name, param in global_model['rep'].named_parameters():
                                    if param.grad is not None:
                                        param.grad.data.div_(total_norm)
                                for name, param in global_model[task].named_parameters():
                                    if param.grad is not None:
                                        param.grad.data.div_(total_norm)

                            local_update_counter += 1

                    for name, param in global_model['rep'].named_parameters():
                        if self.updates[task]['rep'][name] is None:
                            self.updates[task]['rep'][name] = param.grad.data.to(return_device)/config['hyperparameters']['local_training'][
                            'nb_of_local_rounds']
                        else:
                            self.updates[task]['rep'][name] += param.grad.data.to(return_device)/config['hyperparameters']['local_training'][
                            'nb_of_local_rounds']

                    for name, param in global_model[task].named_parameters():
                        if self.updates[task][task][name] is None:
                            self.updates[task][task][name] = param.grad.data.to(return_device)/config['hyperparameters']['local_training'][
                            'nb_of_local_rounds']
                        else:
                            self.updates[task][task][name] += param.grad.data.to(return_device)/config['hyperparameters']['local_training'][
                            'nb_of_local_rounds']
                    # Reset global model to initial state before starting next task training
                    dict_to_model(global_model['rep'], initial_model)
                    dict_to_model(global_model[task], initial_task_model)

                function_return = {"updates": self.updates}
            else:
                current_weight = {task: float(1 / len(tasks)) for i, task in enumerate(tasks)}
                optimizer_last = self.get_optimizer(config, kwargs['last_model'])
                self.grad_saving_device = {True: 'cuda', False: model_device}[boost_w_gpu and kwargs['save_to_gpu']]
                for m in global_model:
                    global_model[m].to(self.grad_saving_device)

                # #### myDel(self.initial_round_gradients)
                if self.grad_saving_device == 'cuda':
                    myAttrDel(self, 'initial_round_gradients')

                [reset_gradients(m) for m in [global_model['rep']] + [global_model[task] for task in tasks]]
                # myDel()
                # Continue with remaining local rounds

                updates = {t: {'rep': None, t: None} for t in tasks}
                for temp in updates.keys():
                    updates[temp]['rep'] = None

                # for task in tasks:
                #     optimizer = self.get_optimizer(config, global_model)
                #     initial_model = model_to_dict(global_model['rep'])
                #     initial_task_model = model_to_dict(global_model[task])
                #
                #     local_update_counter = 0
                #     local_updates_finished_flag = False
                #     while not local_updates_finished_flag:
                #         for batch in self.dataloader:
                #             optimizer.zero_grad()
                #             if local_update_counter == config['hyperparameters']['local_training'][
                #                 'nb_of_local_rounds']:
                #                 local_updates_finished_flag = True
                #                 break
                #
                #             images = experiment_module.trainLoopPreprocess(
                #                 batch[0].to(device))  # if device != config['data']['trainset_device'] else batch[0])
                #             labels = batch[tasks.index(task) + 1].to(
                #                 device)  # if device != config['data']['trainset_device'] else batch[tasks.index(task) + 1]
                #
                #             rep, _ = global_model['rep'](images, None)
                #             out, _ = global_model[task](rep, None)
                #             loss = loss_fn[task](out, labels)
                #             loss.backward()
                #
                #             # 计算上轮次模型的梯度
                #             rep, _ = kwargs['last_model']['rep'](images, None)
                #             out, _ = kwargs['last_model'][task](rep, None)
                #             loss = loss_fn[task](out, labels)
                #             loss.backward()
                #
                            # # config['algorithm_args'][config['algorithm']]['beta'] == 1 / ((kwargs['T'] + 1) ** (2 / 3))
                            # for param, last_param, d in zip(global_model['rep'].parameters(),
                            #                                 kwargs['last_model']['rep'].parameters(),
                            #                                 list(kwargs['last_updates'][task]['rep'].values())):
                            #     param.grad = param.grad + (
                            #             1 - config['algorithm_args'][config['algorithm']]['beta']) * (
                            #                              d - last_param.grad)
                            #
                            #     param.grad = param.grad * (1/len(tasks))
                            #
                            # for param, last_param, d in zip(global_model[task].parameters(),
                            #                                 kwargs['last_model'][task].parameters(),
                            #                                 list(kwargs['last_updates'][task][task].values())):
                            #     param.grad = param.grad + (
                            #             1 - config['algorithm_args'][config['algorithm']]['beta']) * (
                            #                              d - last_param.grad)
                            #
                            #     param.grad = param.grad * (1/len(tasks))
                #
                #             # Normalize gradients if required
                #             if config['algorithm_args'][config['algorithm']]['normalize_local_iters']:
                #                 total_norm = 0.0
                #                 for name, param in global_model['rep'].named_parameters():
                #                     if param.grad is not None:
                #                         total_norm += param.grad.data.norm(2).item() ** 2
                #                 for name, param in global_model[task].named_parameters():
                #                     if param.grad is not None:
                #                         total_norm += param.grad.data.norm(2).item() ** 2
                #                 total_norm = total_norm ** 0.5
                #
                #                 # Normalize gradients
                #                 for name, param in global_model['rep'].named_parameters():
                #                     if param.grad is not None:
                #                         param.grad.data.div_(total_norm)
                #                 for name, param in global_model[task].named_parameters():
                #                     if param.grad is not None:
                #                         param.grad.data.div_(total_norm)
                #
                #             optimizer.step()
                #             local_update_counter += 1
                #
                #     with torch.no_grad():
                #         final_model = model_to_dict(global_model['rep'])
                #         final_task_model = model_to_dict(global_model[task])
                #         [reset_gradients(m) for m in [global_model['rep'], global_model[task]]]
                #         [reset_gradients(m) for m in [kwargs['last_model']['rep'], kwargs['last_model'][task]]]
                #
                #         updates[task]['rep'] = {name: (final_model[name] - initial_model[name]).to(return_device) for
                #                                 name
                #                                 in final_model}
                #         updates[task][task] = {
                #             name: (final_task_model[name] - initial_task_model[name]).to(return_device)
                #             for name in final_task_model}
                #
                #         # Reset global model to initial state before starting next task training
                #         dict_to_model(global_model['rep'], initial_model)
                #         dict_to_model(global_model[task], initial_task_model)
                local_updates_finished_flag, local_update_counter = False, 0
                initial_model = model_to_dict(global_model['rep'])
                initial_task_model = {task:model_to_dict(global_model[task]) for task in tasks}

                while not local_updates_finished_flag:
                    for batch in self.dataloader:

                        optimizer.zero_grad()
                        if local_update_counter == config['hyperparameters']['local_training'][
                            'nb_of_local_rounds']:
                            local_updates_finished_flag = True
                            break

                        task_gradients = {task: {'rep': [], 'task': []} for task in tasks}
                        if device == 'cuda':
                            torch.cuda.empty_cache()

                        for task in tasks:
                            images = experiment_module.trainLoopPreprocess(batch[0].to(
                                device))  # if device != config['data']['trainset_device'] else batch[0])
                            labels = batch[tasks.index(task) + 1].to(
                                device)  # if device != config['data']['trainset_device'] else batch[tasks.index(task) + 1]
                            rep, _ = global_model['rep'](images, None)
                            out, _ = global_model[task](rep, None)
                            loss = loss_fn[task](out, labels)
                            # Zero gradients for this task
                            optimizer.zero_grad()
                            # Compute gradients for the task
                            loss.backward(retain_graph=True)

                            # 计算上轮次模型的梯度
                            rep, _ = kwargs['last_model']['rep'](images, None)
                            out, _ = kwargs['last_model'][task](rep, None)
                            loss = loss_fn[task](out, labels)
                            # Zero gradients for this task
                            optimizer_last.zero_grad()
                            loss.backward()

                            # config['algorithm_args'][config['algorithm']]['beta'] == 1 / ((kwargs['T'] + 1) ** (2 / 3))
                            for param, last_param, d in zip(global_model['rep'].parameters(),
                                                            kwargs['last_model']['rep'].parameters(),
                                                            list(kwargs['last_updates'][task]['rep'].values())):
                                param.grad = param.grad + (
                                        1 - config['algorithm_args'][config['algorithm']]['beta']) * (
                                                     d - last_param.grad)

                            for param, last_param, d in zip(global_model[task].parameters(),
                                                            kwargs['last_model'][task].parameters(),
                                                            list(kwargs['last_updates'][task][task].values())):
                                param.grad = param.grad + (
                                        1 - config['algorithm_args'][config['algorithm']]['beta']) * (
                                                     d - last_param.grad)

                            # Store gradients for 'rep' and task model using state_dict
                            for name, param in global_model['rep'].state_dict(keep_vars=True).items():
                                if param.grad is not None:
                                    # task_gradients[task]['rep'][name] = param.grad.data.clone()
                                    task_gradients[task]['rep'].append(param.grad.data.clone())

                            for name, param in global_model[task].state_dict(keep_vars=True).items():
                                if param.grad is not None:
                                    task_gradients[task]['task'].append(param.grad.data.clone())

                            # 更新累计梯度
                            with torch.no_grad():
                                if local_update_counter == config['hyperparameters']['local_training'][
                                    'nb_of_local_rounds'] - 1:
                                    optimizer.step()
                                    final_model = model_to_dict(global_model['rep'])
                                    final_task_model = model_to_dict(global_model[task])
                                    [reset_gradients(m) for m in [global_model['rep'], global_model[task]]]
                                    [reset_gradients(m) for m in
                                     [kwargs['last_model']['rep'], kwargs['last_model'][task]]]

                                    updates[task]['rep'] = {
                                        name: (final_model[name] - initial_model[name]).to(return_device) for
                                        name
                                        in final_model}
                                    updates[task][task] = {
                                        name: (final_task_model[name] - initial_task_model[task][name]).to(return_device)
                                        for name in final_task_model}
                                else:
                                    continue

                        # Reset gradients after saving them
                        optimizer.zero_grad()
                        [reset_gradients(global_model[t]) for t in global_model]

                        # Normalize gradients if required
                        if config['algorithm_args'][config['algorithm']]['normalize_updates']:
                            for task in tasks:
                                # Compute L2 norm
                                total_norm = 0.0
                                for grad in task_gradients[task]['rep']:
                                    total_norm += grad.norm(2).item() ** 2
                                if config['algorithm_args'][config['algorithm']]['count_decoders']:
                                    for grad in task_gradients[task]['task']:
                                        total_norm += grad.norm(2).item() ** 2
                                total_norm = total_norm ** 0.5
                                # Normalize gradients
                                for grad in task_gradients[task]['rep']:
                                    grad.div_(total_norm)
                                for grad in task_gradients[task]['task']:
                                    grad.div_(total_norm)

                        # Apply weighted gradients
                        for task in tasks:
                            for param, grad in zip(global_model['rep'].parameters(), task_gradients[task]['rep']):
                                if param.grad is None:
                                    param.grad = grad * current_weight[task]
                                else:
                                    param.grad += grad * current_weight[task]
                            temp = current_weight[task] if config['algorithm_args'][config['algorithm']][
                                'scale_decoders'] else 1
                            for param, grad in zip(global_model[task].parameters(), task_gradients[task]['task']):
                                if param.grad is None:
                                    param.grad = grad * temp
                                else:
                                    param.grad += grad * temp
                        optimizer.step()
                        local_update_counter += 1

                function_return = {"updates": updates}

        # 在训练结束后记录性能
        if algorithm in ['fedcmoo', 'fsmgda_vr']:

            # 计算损失减少和准确率提升
            loss_reduction = start_loss - end_loss
            acc_improvement = end_acc - start_acc

            # 存储训练进度
            self.training_progress[algorithm]['loss_reductions'].append(loss_reduction)
            self.training_progress[algorithm]['acc_improvements'].append(acc_improvement)

            # 返回训练进度数据
            if isinstance(function_return, dict):
                function_return['training_progress'] = {
                    'loss_reduction': loss_reduction,
                    'acc_improvement': acc_improvement,
                    'start_loss': start_loss,
                    'start_acc': start_acc,
                    'end_loss': end_loss,
                    'end_acc': end_acc
                }

        return function_return

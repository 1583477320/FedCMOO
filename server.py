from client import Client
import logging
import importlib
import numpy as np
import time
import math
import random
import sys
import os
from threading import Thread
import torch
from queue import PriorityQueue
from threading import Thread
import copy
from torch.utils.data import Subset
import torch.nn as nn
from utils import *
from metrics import get_metrics
from config import Config
from epo_lp import *
import re
from typing import Dict, List, OrderedDict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class Server(object):
    """Multi-objective federated learning server."""

    def __init__(self, config):
        if isinstance(config, str):  # Path is provided
            config = Config(config)
            self.config = config.config
        elif isinstance(config, dict):  # Path is provided
            self.config = config
        else:
            self.config = config.config

        # You can re-load a previously interrupted experiments 
        # Not working with new features TO DO: adapt it to new features, e.g. WandB
        # First, reload that experiment's config file
        self.experiment_reloaded = self.config['reload_exp']['flag']
        if self.config['reload_exp']['flag'] is True:
            reload_config = self.config['reload_exp']
            old_config_file = os.path.join(os.getcwd(),
                                           self.config['paths']['experiment_history'],
                                           self.config['reload_exp']['folder_name'], 'exp_config.json')
            self.config = Config(old_config_file).config
            self.config['reload_exp'] = reload_config

        self.total_clients = self.config["clients"]["total"]

        if self.config['experiment'] == 'QM9':
            from torch_geometric.loader import DataLoader as PyGDataLoader
            self.DataLoader = PyGDataLoader
        else:
            from torch.utils.data import DataLoader as TorchDataLoader
            self.DataLoader = TorchDataLoader

        # WandB initialization
        if self.config['wandb']['flag']:
            self.wandb = __import__('wandb')
            self.wandb.init(project=self.config['wandb']['wandb_project_name'],
                            name=self.config['wandb']['wandb_runname'])

    # Set up server
    def boot(self, use_the_same_dataset_clients=False):

        # Import the experiment module dynamically
        experiment_module_name = self.config["experiment"]
        experiments_path = self.config["paths"]["experiments"]
        # Add experiments directory to the system path
        sys.path.append(experiments_path)
        logging.info(f'Added {experiments_path} to sys.path')
        try:
            self.experiment_module = getattr(__import__(experiment_module_name + '.' + experiment_module_name),
                                             experiment_module_name)
            logging.info(f'Successfully imported module {experiment_module_name}')
        except ImportError as e:
            logging.error(f'Error importing module {experiment_module_name}: {e}')
            raise

        # Create metrics
        self.metrics = get_metrics(self.config, self.experiment_module)

        # Ensure logging is configured
        self.configure_logging()
        logging.info(
            f"You can find experiment results in the folder: {self.config['paths']['experiment_history']}/{self.metrics.exp_id}")

        logging.info('Booting the FL-MOO server...')

        if not use_the_same_dataset_clients:
            # Create clients
            self.create_clients()

            # Load datasets and distribute data to clients
            self.create_dataloaders_distribute_data()

        # Below, it is useful to keep large models in gpu if  model_device == cpu and gpu is available small trick to make faster
        self.boost_w_gpu = True if device == 'cuda' and self.config['model_device'] != 'cuda' else False
        # Load the model
        self.model = self.experiment_module.get_model(self.config)

        # Checkpoint model implementation:
        self.checkPointModel = self.experiment_module.get_model(self.config)
        self.lastTrainMeanLoss = float('inf')
        transfer_parameters(self.model, self.checkPointModel)

        self.model_cuda = {k: copy.deepcopy(self.model[k]).to('cuda') for k in
                           self.model.keys()} if self.boost_w_gpu else None

        # debug gpu use
        logging.info(f"Device: {device}, boost_w_gpu: {self.boost_w_gpu}, model_device: {self.config['model_device']}")

        # Reload model if continuing a previous experiment
        if self.experiment_reloaded:
            self.metrics.load_model(self.model)

            if self.boost_w_gpu:
                transfer_parameters(self.model, self.model_cuda)

            logging.info(f'Training will be resumed from round {self.metrics.current_round}')
            if self.config["hyperparameters"]["local_training"]["local_lr_scheduler_flag"]:  # LR scheduler
                self.config["hyperparameters"]["local_training"]["initial_local_lr"] *= (
                        0.5 ** (self.metrics.current_round // (self.config['max_round'] // 5)))

        # Get tasks
        self.tasks = self.experiment_module.get_tasks()

        if self.config['algorithm'] == 'fedcmoo_pref':
            self.preference = preference = np.array([1 for _ in self.tasks]) if \
                self.config['algorithm_args'][self.config['algorithm']]['preference'] == 'uniform' else np.array(
                self.config['algorithm_args'][self.config['algorithm']]['preference'])
            logging.info(f"The preference is: " + ', '.join(
                [f'{self.tasks[i]}: {temp:.4f}' for i, temp in enumerate(preference)]))

        self.scales = {task: float(1 / len(self.tasks)) for i, task in enumerate(self.tasks)}

        # 初始化c_global和g_global，暂时取0
        if self.config['algorithm'] == 'fedadam':
            self.c_local: Dict[List[torch.Tensor]] = {i: [torch.zeros_like(param).to(device)
                                                          for param in self.model['rep'].parameters()] for i in
                                                      range(len(self.clients))}

            for i, client in enumerate(self.clients):
                if self.boost_w_gpu and 3 * return_free_gpu_memory() > 2 * last_free_memory + 2500:  # Use gpu_save if there is enough memory 500 MB buffer. 3x and 2x are important calculated numbers
                    save_to_gpu = True
                    last_free_memory = return_free_gpu_memory()
                else:
                    save_to_gpu = False
                initial_model = copy.deepcopy(self.model)
                c_local_initial = client.local_train(self.config,
                                                     {key: copy.deepcopy(
                                                         {True: self.model_cuda, False: self.model}[
                                                             self.boost_w_gpu][key]) for key in initial_model},
                                                     self.experiment_module, self.tasks,
                                                     first_local_round=False, initial_c_local=True,
                                                     current_weight=self.scales,
                                                     save_to_gpu=save_to_gpu)
                # 更新c_local
                self.c_local[i] = c_local_initial

            # 初始化 c_global
            self.c_global = [
                torch.zeros_like(param).to(device)
                for param in self.model['rep'].parameters()]

            self.avg_weight = torch.tensor(
                [
                    1 / self.config['clients']['total']
                    for _ in range(self.config['clients']['total'])
                ],
                device=device,
            )
            c_local_list = list(self.c_local.values())
            for c_g, c_del in zip(self.c_global, zip(*c_local_list)):
                c_del = torch.sum(self.avg_weight * torch.stack(c_del, dim=-1), dim=-1)
                c_g.data += c_del

            # 初始化 g_global
            self.g_global = self.c_global

        # 初始化fsmgda-vr的参数
        if self.config['algorithm'] == 'fsmgda_vr':
            self.last_model = copy.deepcopy({True: self.model_cuda, False: self.model}[self.boost_w_gpu])

            averaged_updates = {task: {'rep': {}, task: {}} for task in self.tasks}
            # last_model_recoder = copy.deepcopy({True: self.model_cuda, False: self.model}[self.boost_w_gpu])

            for task in self.tasks:
                # Initialize the 'rep' part using state_dict
                for key, param in self.model['rep'].state_dict().items():
                    averaged_updates[task]['rep'][key] = torch.zeros_like(param, device=device)

                # Initialize the task-specific part using state_dict
                for key, param in self.model[task].state_dict().items():
                    averaged_updates[task][task][key] = torch.zeros_like(param, device=device)
            for i, client in enumerate(self.clients):
                function = client.local_train(self.config,
                                              {key: copy.deepcopy(
                                                  {True: self.model_cuda, False: self.model}[self.boost_w_gpu][key])
                                                  for key in self.model},
                                              self.experiment_module, self.tasks,
                                              initial_d=True
                                              )
                if self.config["algorithm_args"][self.config["algorithm"]]["compression"]:
                    compression_rate = (self.config["proposed_approx_extra_upload_d"] + 1) / len(self.tasks)
                    if compression_rate < 1:
                        function['updates'] = top_k_compression_dict(function['updates'],
                                                                     compression_rate=compression_rate)
                averaged_updates = update_average(averaged_updates, function['updates'], self.tasks,
                                                  1 / len(self.clients))
            self.last_updates = averaged_updates

        if self.config['algorithm'] in ['fedcmoo', 'fedadam', 'fedcmoo_pref']:
            if 'randsvd' in self.config['proposed_approx_method']:
                if 'direct' in self.config['proposed_approx_method']:
                    logging.info(
                        f"To approximate covariance of Jacobian matrix, one-way rand.svd-compressed matrices (with extra {self.config['proposed_approx_extra_upload_d']} d upload) is used.")
                else:
                    logging.info(
                        f"To approximate covariance of Jacobian matrix, our proposed two-way rand.svd-compressed matrices (with extra {self.config['proposed_approx_extra_upload_d']} d upload) is used.")
            elif 'topk' in self.config['proposed_approx_method']:
                if 'direct' in self.config['proposed_approx_method']:
                    logging.info(
                        f"To approximate covariance of Jacobian matrix, one-way top-K-compressed matrices (with extra {self.config['proposed_approx_extra_upload_d']} d upload) is used.")
                else:
                    logging.info(
                        f"To approximate covariance of Jacobian matrix, our proposed two-way top-K-compressed matrices (with extra {self.config['proposed_approx_extra_upload_d']} d upload) is used.")
            else:
                raise Exception("!! Error: Unknown method for approximation of covariance of Jacobian matrix !!")

        if self.config['wandb']['flag']:
            self.wandb.config.update(self.config)

    def train(self):
        """Train the global model using federated learning."""
        # local_lr = self.config["hyperparameters"]["local_training"]["local_lr"]
        
        for m in self.model:
            if self.config['model_device'] == 'cuda' and device == 'cuda':
                self.model[m] = self.model[m].cuda()
            self.model[m].train()
            if self.boost_w_gpu:
                self.model_cuda[m].train()

        start_round = self.metrics.current_round
        for self.round_num in range(start_round, self.config['max_round']):
            if self.config["hyperparameters"]["local_training"]["local_lr_scheduler_flag"]:  # LR scheduler
                # Check if the current round is a multiple of the decay interval
                if self.round_num % 25 == 0 and self.round_num != 0 :
                # if self.round_num % (self.config['max_round'] // 5) == 0 and self.round_num != 0:
                    # Halve the learning rate
                    new_lr = self.config["hyperparameters"]["local_training"]["local_lr"] * 0.5
                    self.config["hyperparameters"]["local_training"]["local_lr"] = new_lr
                    logging.info(f"Round {self.round_num}: Adjusting learning rate to {new_lr:.6f}")
                # elif 'fsmgda_vr' in self.config['algorithm'] :
                # new_lr = local_lr * 0.95 ** self.round_num
                # self.config["hyperparameters"]["local_training"]["local_lr"] = new_lr
                # logging.info(f"Round {self.round_num}: Adjusting learning rate to {new_lr:.6f}")

            starting_time = time.time()

            participating_clients = random.sample(self.clients,
                                                  min(self.config['nb_of_participating_clients'], len(self.clients)))
            algorithm_specific_log = ''
            if self.round_num == 0:
                if len(participating_clients) < self.config['nb_of_participating_clients']:
                    logging.warning(
                        f"Number of total clients ({self.total_clients}) is less than the desired number of participating clients ({self.config['nb_of_participating_clients']}). All clients are participating!")
                else:
                    logging.warning(
                        f"Out of {self.total_clients} clients {self.config['nb_of_participating_clients']} clients participate every round!")

            if self.config['algorithm'] == 'fsmgda':

                client_return_device = 'cuda' if (self.config['model_device'] == 'cuda' or self.boost_w_gpu) else 'cpu'
                # Initialize averaged_updates
                averaged_updates = {task: {'rep': {}, task: {}} for task in self.tasks}
                for task in self.tasks:
                    # Initialize the 'rep' part using state_dict
                    for key, param in self.model['rep'].state_dict().items():
                        averaged_updates[task]['rep'][key] = torch.zeros_like(param, device=client_return_device)

                    # Initialize the task-specific part using state_dict
                    for key, param in self.model[task].state_dict().items():
                        averaged_updates[task][task][key] = torch.zeros_like(param, device=client_return_device)

                for i, client in enumerate(participating_clients):
                    updates = client.local_train(self.config,
                                                 {key: copy.deepcopy(
                                                     {True: self.model_cuda, False: self.model}[self.boost_w_gpu][key])
                                                     for key in self.model},
                                                 self.experiment_module, self.tasks)
                    if self.config["algorithm_args"][self.config["algorithm"]]["compression"]:
                        compression_rate = (self.config["proposed_approx_extra_upload_d"] + 1) / len(self.tasks)
                        if compression_rate < 1:
                            updates = top_k_compression_dict(updates, compression_rate=compression_rate)
                    averaged_updates = update_average(averaged_updates, updates, self.tasks,
                                                      1 / len(participating_clients))

                # Normalize updates
                averaged_updates = normalize_updates(averaged_updates, self.tasks, self.config)

                # Convert updates to vectors
                task_vectors = []
                for task in self.tasks:
                    combined_vector = []
                    if self.config['algorithm_args']['fsmgda']['count_decoders']:
                        for key in averaged_updates[task]['rep']:
                            combined_vector.append(averaged_updates[task]['rep'][key].view(-1))
                        for key in averaged_updates[task][task]:
                            combined_vector.append(averaged_updates[task][task][key].view(-1))
                    else:
                        for key in averaged_updates[task]['rep']:
                            combined_vector.append(averaged_updates[task]['rep'][key].view(-1))
                    task_vectors.append(torch.cat(combined_vector).reshape(1, -1))

                # Frank-Wolfe iteration to compute scales
                try:
                    sol, min_norm = MinNormSolver.find_min_norm_element(task_vectors)
                except:
                    logging.info('\nException: MinNormSolver failed!\n')
                    sol = [1 / len(self.tasks) for _ in self.tasks]
                self.scales = {task: float(sol[i]) for i, task in enumerate(self.tasks)}

                # Log task name and weight scale for each task
                algorithm_specific_log += ' Scales: ' + ', '.join(
                    [f'{task}: {self.scales[task]:.4f}' for task in self.scales])

                # Aggregate updates to update the global model
                self.aggregate_updates(model_to_aggregate={True: self.model_cuda, False: self.model}[self.boost_w_gpu],
                                       normalized_updates=averaged_updates, scales=self.scales)
                if self.boost_w_gpu:
                    transfer_parameters(self.model_cuda, self.model)

            elif self.config['algorithm'] == 'fedcmoo':
                scale_updates = []
                last_free_memory = 0
                for i, client in enumerate(participating_clients):
                    if self.boost_w_gpu and 3 * return_free_gpu_memory() > 2 * last_free_memory + 2500:  # Use gpu_save if there is enough memory 500 MB buffer. 3x and 2x are important calculated numbers
                        save_to_gpu = True
                        last_free_memory = return_free_gpu_memory()
                    else:
                        save_to_gpu = False
                    client_scale_update, _ = client.local_train(self.config,
                                                                {key: copy.deepcopy(
                                                                    {True: self.model_cuda, False: self.model}[
                                                                        self.boost_w_gpu][key]) for key in self.model},
                                                                self.experiment_module, self.tasks,
                                                                first_local_round=True, current_weight=self.scales,
                                                                save_to_gpu=save_to_gpu)
                    scale_updates.append(client_scale_update)

                G_T_G_estimate = self.estimateG_T_G(matrices=scale_updates,
                                                    method=self.config['proposed_approx_method']) / len(scale_updates)

                # Update idea adapted from https://github.com/OptMN-Lab/sdmgrad/blob/main/methods/weight_methods.py#L770
                self.fedcmoo_update_scales(torch.tensor(G_T_G_estimate))
                algorithm_specific_log += f' Scales: ' + ', '.join(
                    [f'{task}: {self.scales[task]:.4f}' for task in self.scales])

                # Now continue the remaining local rounds
                # Initialize averaged_updates
                averaged_updates = {'rep': {}, **{task: {} for task in self.tasks}}

                # Initialize 'rep' part using state_dict
                for key, param in {True: self.model_cuda, False: self.model}[self.boost_w_gpu][
                    'rep'].state_dict().items():
                    averaged_updates['rep'][key] = torch.zeros_like(param)

                # Initialize task-specific parts using state_dict
                for task in self.tasks:
                    for key, param in {True: self.model_cuda, False: self.model}[self.boost_w_gpu][
                        task].state_dict().items():
                        averaged_updates[task][key] = torch.zeros_like(param)

                for i, client in enumerate(participating_clients):
                    updates = client.local_train(self.config,
                                                 {key: copy.deepcopy(
                                                     {True: self.model_cuda, False: self.model}[self.boost_w_gpu][key])
                                                     for key in self.model},
                                                 self.experiment_module, self.tasks, first_local_round=False,
                                                 current_weight=self.scales)
                    # Apply weighted updates
                    for key in updates['rep']:
                        averaged_updates['rep'][key] += updates['rep'][key] / len(participating_clients)
                    for task in self.tasks:
                        for key in updates[task]:
                            averaged_updates[task][key] += updates[task][key] / len(participating_clients)
                averaged_updates = normalize_updates(averaged_updates, self.tasks, self.config)

                self.aggregate_updates(model_to_aggregate={True: self.model_cuda, False: self.model}[self.boost_w_gpu],
                                       normalized_updates=averaged_updates)
                if self.boost_w_gpu:
                    transfer_parameters(self.model_cuda, self.model)

            elif self.config['algorithm'] == 'fedcmoo_pref':
                scale_updates = []
                last_free_memory = 0
                for i, client in enumerate(participating_clients):
                    if self.boost_w_gpu and 3 * return_free_gpu_memory() > 2 * last_free_memory + 2500:  # Use gpu_save if there is enough memory 500 MB buffer. 3x and 2x are important calculated numbers
                        save_to_gpu = True
                        last_free_memory = return_free_gpu_memory()
                    else:
                        save_to_gpu = False
                    client_scale_update, _ = client.local_train(self.config,
                                                                {key: copy.deepcopy(
                                                                    {True: self.model_cuda, False: self.model}[
                                                                        self.boost_w_gpu][key]) for key in self.model},
                                                                self.experiment_module, self.tasks,
                                                                first_local_round=True, current_weight=self.scales,
                                                                save_to_gpu=save_to_gpu)
                    scale_updates.append(client_scale_update)
                G_T_G_estimate = self.estimateG_T_G(matrices=scale_updates, method=self.config[
                    'proposed_approx_method'])  # scaling doesn't matter
                # myDel(scale_updates)
                average_loss_np = self.getParticipatingClientLoss(participating_clients)

                if not hasattr(self, 'epo'):
                    preference = np.array([1 for _ in self.tasks]) if self.config['algorithm_args']['fedcmoo_pref'][
                                                                          'preference'] == 'uniform' else np.array(
                        self.config['algorithm_args']['fedcmoo_pref']['preference'])
                    if 'epo_eps' in self.config["exp_identifier"]:
                        temp = float(
                            re.search(r'epo_eps(-?\d+\.?\d*(e[+-]?\d+)?)', self.config["exp_identifier"]).group(1))
                        self.epo = EPO_LP(m=len(self.tasks), n=None, r=preference, eps=temp)
                    else:
                        self.epo = EPO_LP(m=len(self.tasks), n=None, r=preference, eps=1e-2)

                scalesArr = self.epo.get_alpha(l=average_loss_np, G=G_T_G_estimate, C=True, relax=True)

                # # sil 4 deneme debug sil
                min_weight_multiplier = self.config["algorithm_args"]["fedcmoo_pref"]["min_weight_multiplier"]
                scalesArr = scalesArr * (1 - min_weight_multiplier) + np.array(
                    [min_weight_multiplier / len(self.tasks) for _ in self.tasks])

                for i, alpha in enumerate(scalesArr):
                    self.scales[self.tasks[i]] = alpha

                algorithm_specific_log += f'Partc. Client Losses: ' + ', '.join(
                    [f'{self.tasks[i]}: {loss:.3f}' for i, loss in
                     enumerate(average_loss_np)]) + f' | Scales: ' + ', '.join(
                    [f'{task}: {self.scales[task]:.3f}' for task in self.scales])

                # Now continue the remaining local rounds
                # Initialize averaged_updates
                averaged_updates = {'rep': {}, **{task: {} for task in self.tasks}}

                # Initialize 'rep' part using state_dict
                for key, param in {True: self.model_cuda, False: self.model}[self.boost_w_gpu][
                    'rep'].state_dict().items():
                    averaged_updates['rep'][key] = torch.zeros_like(param)

                # Initialize task-specific parts using state_dict
                for task in self.tasks:
                    for key, param in {True: self.model_cuda, False: self.model}[self.boost_w_gpu][
                        task].state_dict().items():
                        averaged_updates[task][key] = torch.zeros_like(param)

                for i, client in enumerate(participating_clients):
                    updates = client.local_train(self.config,
                                                 {key: copy.deepcopy(
                                                     {True: self.model_cuda, False: self.model}[self.boost_w_gpu][key])
                                                     for key in self.model},
                                                 self.experiment_module, self.tasks, first_local_round=False,
                                                 current_weight=self.scales)
                    # Apply weighted updates
                    for key in updates['rep']:
                        averaged_updates['rep'][key] += updates['rep'][key] / len(participating_clients)
                    for task in self.tasks:
                        for key in updates[task]:
                            averaged_updates[task][key] += updates[task][key] / len(participating_clients)
                averaged_updates = normalize_updates(averaged_updates, self.tasks, self.config)

                self.aggregate_updates(model_to_aggregate={True: self.model_cuda, False: self.model}[self.boost_w_gpu],
                                       normalized_updates=averaged_updates)
                if self.boost_w_gpu:
                    transfer_parameters(self.model_cuda, self.model)

            elif self.config['algorithm'] == 'fedadam':
                scale_updates = []
                last_free_memory = 0
                for i, client in enumerate(participating_clients):
                    if self.boost_w_gpu and 3 * return_free_gpu_memory() > 2 * last_free_memory + 2500:  # Use gpu_save if there is enough memory 500 MB buffer. 3x and 2x are important calculated numbers
                        save_to_gpu = True
                        last_free_memory = return_free_gpu_memory()
                    else:
                        save_to_gpu = False
                    client_scale_update, _ = client.local_train(self.config,
                                                                {key: copy.deepcopy(
                                                                    {True: self.model_cuda, False: self.model}[
                                                                        self.boost_w_gpu][key]) for key in self.model},
                                                                self.experiment_module, self.tasks,
                                                                first_local_round=True, initial_c_local=False,
                                                                current_weight=self.scales,
                                                                save_to_gpu=save_to_gpu)
                    scale_updates.append(client_scale_update)

                G_T_G_estimate = self.estimateG_T_G(matrices=scale_updates,
                                                    method=self.config['proposed_approx_method']) / len(scale_updates)

                # Update idea adapted from https://github.com/OptMN-Lab/sdmgrad/blob/main/methods/weight_methods.py#L770
                self.fedcmoo_update_scales(torch.tensor(G_T_G_estimate))
                algorithm_specific_log += f' Scales: ' + ', '.join(
                    [f'{task}: {self.scales[task]:.4f}' for task in self.scales])

                # Now continue the remaining local rounds
                # Initialize averaged_updates
                averaged_updates = {'rep': {}, **{task: {} for task in self.tasks}}

                # Initialize 'rep' part using state_dict,初始化averaged_updates
                for key, param in {True: self.model_cuda, False: self.model}[self.boost_w_gpu][
                    'rep'].state_dict().items():
                    averaged_updates['rep'][key] = torch.zeros_like(param)

                # Initialize task-specific parts using state_dict
                for task in self.tasks:
                    for key, param in {True: self.model_cuda, False: self.model}[self.boost_w_gpu][
                        task].state_dict().items():
                        averaged_updates[task][key] = torch.zeros_like(param)

                for i, client in enumerate(participating_clients):
                    updates = client.local_train(self.config,
                                                 {key: copy.deepcopy(
                                                     {True: self.model_cuda, False: self.model}[self.boost_w_gpu][key])
                                                     for key in self.model},
                                                 self.experiment_module, self.tasks, first_local_round=False,
                                                 initial_c_local=False,
                                                 current_weight=self.scales,
                                                 c_global=self.c_global,
                                                 g_global=self.g_global,
                                                 c_local=self.c_local[i]
                                                 )
                    # 更新c_local
                    self.c_local[i] = updates['c_local']

                    # Apply weighted updates
                    div = (len(participating_clients) * self.config['hyperparameters']['local_training'][
                        'nb_of_local_rounds'] * self.config['hyperparameters']['local_training']['local_lr'])
                    for key in updates['rep']:
                        averaged_updates['rep'][key] += (updates['rep'][key] / div)
                    for task in self.tasks:
                        for key in updates[task]:
                            averaged_updates[task][key] += (updates[task][key] / div)
                averaged_updates = normalize_updates(averaged_updates, self.tasks, self.config)

                self.aggregate_updates(model_to_aggregate={True: self.model_cuda, False: self.model}[self.boost_w_gpu],
                                       normalized_updates=averaged_updates)
                if self.boost_w_gpu:
                    transfer_parameters(self.model_cuda, self.model)

                # 更新控制变量c和g
                c_clone = copy.deepcopy(self.c_global)
                self.c_aggregate(self.c_local, c_clone)

            elif self.config['algorithm'] == 'fsmgda_vr':

                client_return_device = 'cuda' if (self.config['model_device'] == 'cuda' or self.boost_w_gpu) else 'cpu'
                # Initialize averaged_updates
                averaged_updates = {task: {'rep': {}, task: {}} for task in self.tasks}
                # last_model_recoder = copy.deepcopy({True: self.model_cuda, False: self.model}[self.boost_w_gpu])

                for task in self.tasks:
                    # Initialize the 'rep' part using state_dict
                    for key, param in self.model['rep'].state_dict().items():
                        averaged_updates[task]['rep'][key] = torch.zeros_like(param, device=client_return_device)

                    # Initialize the task-specific part using state_dict
                    for key, param in self.model[task].state_dict().items():
                        averaged_updates[task][task][key] = torch.zeros_like(param, device=client_return_device)

                for i, client in enumerate(participating_clients):
                    function = client.local_train(self.config,
                                                 {key: copy.deepcopy(
                                                     {True: self.model_cuda, False: self.model}[self.boost_w_gpu][key])
                                                     for key in self.model},
                                                 self.experiment_module, self.tasks,
                                                 last_model = self.last_model, # 上批次模型
                                                 last_updates = self.last_updates, # 上批次梯度
                                                 T=self.round_num,
                                                 initial_d=False
                                                 )
                    if self.config["algorithm_args"][self.config["algorithm"]]["compression"]:
                        compression_rate = (self.config["proposed_approx_extra_upload_d"] + 1) / len(self.tasks)
                        if compression_rate < 1:
                            function['updates'] = top_k_compression_dict(function['updates'], compression_rate=compression_rate)
                    averaged_updates = update_average(averaged_updates, function['updates'], self.tasks,
                                                      1 / len(participating_clients))

                # Normalize updates
                averaged_updates = normalize_updates(averaged_updates, self.tasks, self.config)

                # 更新上批次参数
                self.last_updates = averaged_updates
                self.last_model = copy.deepcopy({key: copy.deepcopy(
                                                     {True: self.model_cuda, False: self.model}[self.boost_w_gpu][key])
                                                     for key in self.model})

                # Convert updates to vectors
                task_vectors = []
                for task in self.tasks:
                    combined_vector = []
                    if self.config['algorithm_args']['fsmgda']['count_decoders']:
                        for key in averaged_updates[task]['rep']:
                            combined_vector.append(averaged_updates[task]['rep'][key].view(-1))
                        for key in averaged_updates[task][task]:
                            combined_vector.append(averaged_updates[task][task][key].view(-1))
                    else:
                        for key in averaged_updates[task]['rep']:
                            combined_vector.append(averaged_updates[task]['rep'][key].view(-1))
                    task_vectors.append(torch.cat(combined_vector).reshape(1, -1))

                # Frank-Wolfe iteration to compute scales
                try:
                    sol, min_norm = MinNormSolver.find_min_norm_element(task_vectors)
                except:
                    logging.info('\nException: MinNormSolver failed!\n')
                    sol = [1 / len(self.tasks) for _ in self.tasks]
                self.scales = {task: float(sol[i]) for i, task in enumerate(self.tasks)}

                # Log task name and weight scale for each task
                algorithm_specific_log += ' Scales: ' + ', '.join(
                    [f'{task}: {self.scales[task]:.4f}' for task in self.scales])

                # Aggregate updates to update the global model
                self.aggregate_updates(model_to_aggregate={True: self.model_cuda, False: self.model}[self.boost_w_gpu],
                                       normalized_updates=averaged_updates, scales=self.scales)

                if self.boost_w_gpu:
                    transfer_parameters(self.model_cuda, self.model)



            # Initialize an empty dictionary to collect all WandB logs
            wandb_log_data = {}

            # Evaluate train, validation, and testing metrics
            if self.config['metrics']['train_period'] > 0 and (self.round_num + 1) % self.config['metrics'][
                'train_period'] == 0:
                client_total_metrics = [self.evaluate_metrics('train', client=c) for c in self.clients]
                average_total_metrics = client_total_metrics[0].copy()
                for task in self.tasks:
                    for metric in self.metrics.eval_metrics[task]:
                        average_total_metrics[task][metric] = sum(
                            [client_total_metrics[i][task][metric] for i in range(len(participating_clients))]) / len(
                            participating_clients)

                # Checkpoint model
                loss_values = [average_total_metrics[t]['loss'] for t in self.tasks if
                               'loss' in self.metrics.eval_metrics[t]]
                has_invalid_loss = any(math.isnan(loss) or math.isinf(loss) for loss in loss_values)
                if has_invalid_loss:  # Model divergence happens
                    transfer_parameters(self.checkPointModel, self.model)
                    if self.boost_w_gpu:
                        transfer_parameters(self.checkPointModel, self.model_cuda)
                    client_total_metrics = [self.evaluate_metrics('train', client=c) for c in self.clients]
                    average_total_metrics = client_total_metrics[0].copy()
                    for task in self.tasks:
                        for metric in self.metrics.eval_metrics[task]:
                            average_total_metrics[task][metric] = sum([client_total_metrics[i][task][metric] for i in
                                                                       range(len(participating_clients))]) / len(
                                participating_clients)
                else:
                    mean_loss = sum(loss_values) / len(loss_values) if loss_values else None
                    if mean_loss:
                        if mean_loss > 1.5 * self.lastTrainMeanLoss:  # Model divergence happens
                            transfer_parameters(self.checkPointModel, self.model)
                            if self.boost_w_gpu:
                                transfer_parameters(self.checkPointModel, self.model_cuda)
                            client_total_metrics = [self.evaluate_metrics('train', client=c) for c in self.clients]
                            average_total_metrics = client_total_metrics[0].copy()
                            for task in self.tasks:
                                for metric in self.metrics.eval_metrics[task]:
                                    average_total_metrics[task][metric] = sum(
                                        [client_total_metrics[i][task][metric] for i in
                                         range(len(participating_clients))]) / len(participating_clients)
                        else:  # normal training, update checkpoint model
                            transfer_parameters(self.model, self.checkPointModel)
                            self.lastTrainMeanLoss = mean_loss

                self.metrics.update_metrics('train', average_total_metrics)
                log_message = f'Round {self.round_num + 1}: Train eval:'
                for task in self.tasks:
                    for metric in self.metrics.eval_metrics[task]:
                        log_message += f'Task {task} {metric} = {average_total_metrics[task][metric]:.4f}, '
                logging.info(log_message.rstrip(', '))

                # Log to WandB
                if self.config['wandb']['flag']:
                    # Log each task's metrics individually
                    for task in self.tasks:
                        for metric in self.metrics.eval_metrics[task]:
                            wandb_log_data[f'train_{task}_{metric}'] = average_total_metrics[task][metric]
                    # Compute mean and std across tasks for all metrics
                    for metric in set([metric for task in self.tasks for metric in self.metrics.eval_metrics[task]]):
                        task_values = [average_total_metrics[task][metric] for task in self.tasks if
                                       metric in self.metrics.eval_metrics[task]]
                        if task_values:  # Only log if there are valid tasks with this metric
                            wandb_log_data[f'train_mean_{metric}'] = np.mean(task_values)
                            wandb_log_data[f'train_std_{metric}'] = np.std(task_values)

            if self.config['metrics']['val_period'] > 0 and self.val_loader and (self.round_num + 1) % \
                    self.config['metrics']['val_period'] == 0:
                average_total_metrics = self.evaluate_metrics('val')
                self.metrics.update_metrics('val', average_total_metrics)
                log_message = f'Round {self.round_num + 1}: Val eval: '
                for task in self.tasks:
                    for metric in self.metrics.eval_metrics[task]:
                        log_message += f'Task {task} {metric} = {average_total_metrics[task][metric]:.4f}, '
                logging.info(log_message.rstrip(', '))
                # Log to WandB
                if self.config['wandb']['flag']:
                    # Log each task's metrics individually
                    for task in self.tasks:
                        for metric in self.metrics.eval_metrics[task]:
                            wandb_log_data[f'val_{task}_{metric}'] = average_total_metrics[task][metric]

                    # Compute mean and std across tasks for all metrics
                    for metric in set([metric for task in self.tasks for metric in self.metrics.eval_metrics[task]]):
                        task_values = [average_total_metrics[task][metric] for task in self.tasks if
                                       metric in self.metrics.eval_metrics[task]]
                        if task_values:  # Only log if there are valid tasks with this metric
                            wandb_log_data[f'val_mean_{metric}'] = np.mean(task_values)
                            wandb_log_data[f'val_std_{metric}'] = np.std(task_values)

            if self.config['metrics']['test_period'] > 0 and (self.round_num + 1) % self.config['metrics'][
                'test_period'] == 0:
                average_total_metrics = self.evaluate_metrics('test')
                self.metrics.update_metrics('test', average_total_metrics)
                log_message = f'Round {self.round_num + 1}: Test eval: '
                for task in self.tasks:
                    for metric in self.metrics.eval_metrics[task]:
                        log_message += f'Task {task} {metric} = {average_total_metrics[task][metric]:.4f}, '
                logging.info(log_message.rstrip(', '))
                # Log to WandB
                if self.config['wandb']['flag']:
                    # Log each task's metrics individually
                    for task in self.tasks:
                        for metric in self.metrics.eval_metrics[task]:
                            wandb_log_data[f'test_{task}_{metric}'] = average_total_metrics[task][metric]

                    # Compute mean and std across tasks for all metrics
                    for metric in set([metric for task in self.tasks for metric in self.metrics.eval_metrics[task]]):
                        task_values = [average_total_metrics[task][metric] for task in self.tasks if
                                       metric in self.metrics.eval_metrics[task]]
                        if task_values:  # Only log if there are valid tasks with this metric
                            wandb_log_data[f'test_mean_{metric}'] = np.mean(task_values)
                            wandb_log_data[f'test_std_{metric}'] = np.std(task_values)

            # Log everything to WandB at once, if there are logs to report
            if self.config['wandb']['flag'] and wandb_log_data:
                # Add the round number
                wandb_log_data['round'] = self.round_num + 1
                self.wandb.log(wandb_log_data)

            if self.config['metrics']['model_save_period'] > 0 and (self.round_num + 1) % self.config['metrics'][
                'model_save_period'] == 0:
                logging.info(f'Round {self.round_num + 1}: Models are saved!')
                self.metrics.save_model(self.model)

            logging.info(
                f'Completed round {self.round_num + 1}/{self.config["max_round"]} in {time.time() - starting_time:.1f} seconds! ' + algorithm_specific_log)
        logging.info('Training completed.')

    # Server - helper functions
    def create_clients(self):
        """Create clients based on the configuration."""
        self.clients = []
        logging.info(f'Creating {self.total_clients} clients...')
        for i in range(self.total_clients):
            client = Client(client_id=i)
            self.clients.append(client)
        logging.info(f'Successfully created {len(self.clients)} clients.')

    def create_dataloaders_distribute_data(self):
        """Load and distribute data to the clients."""
        logging.info('Loading data...')

        # Load training data
        if self.config['experiment'] == "QM9":

            if self.config['data']['distribution'] != 'uniform':
                logging.info(
                    "QM9 experiment is only supported with uniform molecule-wise distribution. Configuration is set to uniform.")
                self.config['data']['distribution'] = 'uniform'
            train_dataset = self.experiment_module.Dataset(split='train', datasetdevice=self.config['data'][
                'trainset_device']).qm9_dataset
        else:
            train_dataset = self.experiment_module.Dataset(split='train',
                                                           datasetdevice=self.config['data']['trainset_device'],
                                                           pre_transform=self.config['data']['pre_transform'])

        # Check if validation split is needed
        val_ratio = self.config['data']['val_ratio']
        if val_ratio > 0 and val_ratio < 1:
            if self.config["experiment"] in ["MultiMNIST", "CIFAR10_MNIST", "MNIST_FMNIST"]:
                train_dataset, val_dataset = self.experiment_module.split_val_dataset(train_dataset, val_ratio,
                                                                                      val_seed=self.config['data'][
                                                                                          'val_seed'])
            elif self.config['experiment'] in ['CelebA', 'CelebA_CNN', 'CelebA5', "CelebA5_CNN"]:
                val_dataset = self.experiment_module.Dataset(split='val',
                                                             datasetdevice=self.config['data']['trainset_device'],
                                                             pre_transform=self.config['data']['pre_transform'])
                val_dataset.split = "test"
                val_dataset.train = False
            elif self.config['experiment'] == 'QM9':
                val_dataset = self.experiment_module.Dataset(split='val', datasetdevice=self.config['data'][
                    'trainset_device']).qm9_dataset
        else:
            val_dataset = None

        # Load test data
        if self.config['experiment'] == "QM9":
            test_dataset = self.experiment_module.Dataset(split='test', datasetdevice=self.config['data'][
                'testset_device']).qm9_dataset
        else:
            test_dataset = self.experiment_module.Dataset(split='test',
                                                          datasetdevice=self.config['data']['testset_device'],
                                                          pre_transform=self.config['data']['pre_transform'])

        # Create DataLoaders for validation and test datasets
        self.val_loader = self.DataLoader(val_dataset, batch_size=self.config['data']['test_batch_size'],
                                          shuffle=False) if val_dataset else None
        self.test_loader = self.DataLoader(test_dataset, batch_size=self.config['data']['test_batch_size'],
                                           shuffle=False)

        # Partition the training data among clients
        self.partition_data(train_dataset)

        logging.info('Data loaded and distributed successfully.')

    def partition_data(self, train_dataset):
        """Partition training data among clients."""
        num_data_points = self.config['clients']['data_points']
        distribution = self.config['data']['distribution']
        partition_seed = self.config['clients']['partition_seed']

        # Initialize the random seed
        np.random.seed(partition_seed)

        # Shuffle all data indices
        all_indices = list(range(len(train_dataset)))
        np.random.shuffle(all_indices)

        if distribution == 'uniform':
            # Uniform random partition
            client_indices = [[] for _ in range(self.total_clients)]
            eligible_clients = list(range(self.total_clients))  # Initialize eligible clients

            while eligible_clients:
                for idx in all_indices:
                    if not eligible_clients:
                        break  # No more eligible clients
                    # Randomly choose one of these clients
                    selected_client = np.random.choice(eligible_clients)
                    # Assign the index to the selected client
                    client_indices[selected_client].append(idx)
                    # Check if the selected client has reached the desired number of data points
                    if len(client_indices[selected_client]) >= num_data_points:
                        eligible_clients.remove(selected_client)
                # If we exhaust all indices, reshuffle and continue
                np.random.shuffle(all_indices)

        elif 'dirichlet' in distribution:
            if distribution == 'dirichlet_all_labels':
                labels_unique_cpu = train_dataset.labels_unique.cpu().numpy()
            elif distribution == 'dirichlet_first_label':
                if self.config["experiment"] in ["MultiMNIST", "MNIST_FMNIST"]:
                    labels_unique_cpu = train_dataset.labels_l.cpu().numpy()
                elif self.config["experiment"] in ["CIFAR10_MNIST"]:
                    labels_unique_cpu = train_dataset.labels_cifar.cpu().numpy()
                elif self.config["experiment"] in ["CelebA", "CelebA_CNN"]:
                    labels_unique_cpu = np.array([int(''.join(map(str, bits)), 2) for bits in train_dataset.labels[:,
                                                                                              [2, 20,
                                                                                               21]].cpu().numpy()])  # These tasks are selected based on the homogeneoity
                elif self.config["experiment"] in ["CelebA5", "CelebA5_CNN"]:
                    labels_unique_cpu = np.array([int(''.join(map(str, bits)), 2) for bits in
                                                  train_dataset.original_labels[:, [2, 20, 21]].cpu().numpy()])
            else:
                logging.warning(f"Unsupported dirichlet distribution: {distribution}")

            # Dirichlet distribution
            client_indices = [[] for _ in range(self.total_clients)]

            label_indices_map = {label: np.where(labels_unique_cpu == label)[0].tolist() for label in
                                 np.unique(labels_unique_cpu)}
            # Shuffle label indices initially
            for label in label_indices_map:
                np.random.shuffle(label_indices_map[label])

            # Store used indices for potential extension
            used_label_indices_map = {label: [] for label in label_indices_map}

            for client_id in range(self.total_clients):
                label_proportions = np.random.dirichlet([self.config['data']['diric_alpha']] * len(label_indices_map))
                for label, proportion in zip(label_indices_map.keys(), label_proportions):
                    num_label_data_points = int(proportion * num_data_points)
                    label_indices = label_indices_map[label]

                    if len(label_indices) < num_label_data_points:
                        # Extend label indices map with used indices
                        used_label_indices_map[label].extend(label_indices)
                        label_indices_map[label] = used_label_indices_map[label]
                        np.random.shuffle(label_indices_map[label])
                        used_label_indices_map[label] = []
                        label_indices = label_indices_map[label]

                    extended_indices = label_indices[:num_label_data_points]
                    used_label_indices_map[label].extend(extended_indices)

                    client_indices[client_id].extend(extended_indices)
                    label_indices_map[label] = label_indices[num_label_data_points:]

                # If client does not have enough samples, assign from unused samples
                while len(client_indices[client_id]) < num_data_points:
                    remaining_indices = num_data_points - len(client_indices[client_id])
                    all_available_indices = [idx for label in label_indices_map for idx in label_indices_map[label]]
                    np.random.shuffle(all_available_indices)
                    for _ in range(remaining_indices):
                        chosen_idx = all_available_indices.pop()
                        chosen_label = int(labels_unique_cpu[chosen_idx])
                        client_indices[client_id].append(chosen_idx)
                        label_indices_map[chosen_label].remove(chosen_idx)
                        used_label_indices_map[chosen_label].append(chosen_idx)

        else:
            raise ValueError(f"Unsupported data distribution: {distribution}")

        # Assign data to clients
        for i, client in enumerate(self.clients):
            if self.config['experiment'] == 'QM9':
                client_data = self.experiment_module.QM9Dataset(train_dataset.rawAll[client_indices[i]],
                                                                [int(i[1:]) for i in
                                                                 self.experiment_module.get_tasks()],
                                                                device=self.config['data']['trainset_device'])
            else:
                client_data = Subset(train_dataset, client_indices[i])
            client.set_data(client_data, self.config)

    def aggregate_updates(self, model_to_aggregate, **kwargs):
        if self.config['algorithm'] in ['fsmgda', 'fsmgda_vr']:
            # Aggregate updates for common part
            scales = kwargs['scales']
            normalized_updates = kwargs['normalized_updates']
            for param_name in model_to_aggregate['rep'].state_dict().keys():
                update_sum = sum(scales[t] * normalized_updates[t]['rep'][param_name] for t in self.tasks)
                model_to_aggregate['rep'].state_dict()[param_name].add_(
                    self.config['hyperparameters']['global_lr'] * update_sum)

            # Update decoder models based on scale_decoders flag
            for task in self.tasks:
                for param_name in model_to_aggregate[task].state_dict().keys():
                    if self.config['algorithm_args'][self.config['algorithm']]['scale_decoders']:
                        update_value = self.config['hyperparameters']['global_lr'] * scales[task] * \
                                       normalized_updates[task][task][param_name]
                    else:
                        update_value = self.config['hyperparameters']['global_lr'] * normalized_updates[task][task][
                            param_name]
                    model_to_aggregate[task].state_dict()[param_name].add_(update_value)
        elif (self.config['algorithm'] in ['fedcmoo', 'fedadam']) or (
                'fedcmoo_pref' in self.config['algorithm']):
            normalized_updates = kwargs['normalized_updates']
            # Update the 'rep' part
            for param_name in model_to_aggregate['rep'].state_dict().keys():
                update_sum = normalized_updates['rep'][param_name]
                model_to_aggregate['rep'].state_dict()[param_name].add_(
                    self.config['hyperparameters']['global_lr'] * update_sum)

            # Update each task model
            for task in self.tasks:
                for param_name in model_to_aggregate[task].state_dict().keys():
                    update_value = normalized_updates[task][param_name]
                    model_to_aggregate[task].state_dict()[param_name].add_(
                        self.config['hyperparameters']['global_lr'] * update_value)

    def evaluate_metrics(self, phase, client=None):
        if phase == 'val' and self.val_loader:
            loader = self.val_loader
        elif phase == 'test':
            loader = self.test_loader
        elif phase == 'train':
            loader = client.dataloader
        else:
            return
        model = self.model_cuda if self.boost_w_gpu else self.model
        for m in model:
            model[m].eval()
        total_metrics = {task: {temp: 0.0 for temp in self.metrics.eval_metrics[task]} for task in self.tasks}

        with torch.no_grad():
            loss_fn = self.experiment_module.get_loss()

            total_samples = 0
            for batch in loader:
                images = batch[0].to(device)  # if device != self.config['data'][f'{phase}set_device'] else batch[0]
                labels = {task: batch[idx + 1].to(device) for idx, task in enumerate(
                    self.tasks)}  # if device != self.config['data'][f'{phase}set_device'] else batch[idx + 1] for idx, task in enumerate(self.tasks)}
                rep, mask = model['rep'](images, None)
                for task in self.tasks:
                    out, _ = model[task](rep, None)

                    for metric in self.metrics.eval_metrics[task]:
                        if metric == 'loss':
                            loss = loss_fn[task](out, labels[task]).item()
                            total_metrics[task]['loss'] += loss * images.size(0)
                        elif metric == 'accuracy':
                            if self.config['experiment'] in ['CelebA5', "CelebA5_CNN"]:
                                predicted = (out > 0.5).float()
                                correct_predictions = (predicted == labels[task]).float().sum().item()
                                total_metrics[task]['accuracy'] += correct_predictions / 8
                            else:
                                accuracy = (out.argmax(dim=1) == labels[task]).float().mean().item()
                                total_metrics[task]['accuracy'] += accuracy * images.size(0)
                total_samples += images.size(0)

            # Normalizing the metrics
            for task in self.tasks:
                for metric in self.metrics.eval_metrics[task]:
                    total_metrics[task][metric] /= total_samples

        for m in model:
            model[m].train()
            # self.model[m].to(self.config['model_device'])
        self.metrics.current_round = self.round_num

        if phase != 'train':
            self.metrics.update_metrics(phase, total_metrics)
        return total_metrics

    def fedcmoo_update_scales(self, GG):
        # Normalize GG
        normalization = torch.mean(torch.sqrt(torch.diag(GG) + 1e-4))
        GG = GG / normalization.pow(2)

        # Vectorized form of self.scales
        w = torch.tensor([self.scales[key] for key in self.tasks], dtype=torch.float32)
        w.requires_grad = True

        # Optimizer setup
        scale_lr = self.config["algorithm_args"]["fedcmoo"]["scale_lr"]
        scale_momentum = self.config["algorithm_args"]["fedcmoo"]["scale_momentum"]
        optimizer = torch.optim.SGD([w], lr=scale_lr, momentum=scale_momentum)

        # Optimization loop
        for _ in range(self.config["algorithm_args"]["fedcmoo"]["scale_n_iter"]):
            optimizer.zero_grad()
            w.grad = torch.mv(GG, w.detach())
            optimizer.step()
            proj = proj_simplex(w.data.cpu().numpy())
            w.data.copy_(torch.from_numpy(proj).data)

        w.requires_grad = False

        # Re-assign new w values to self.scales in the correct order
        for i, key in enumerate(self.tasks):
            self.scales[key] = w[i].item()

    def configure_logging(self):
        """Configure logging to write to a local file and log to the console."""
        logger = logging.getLogger()
        if logger.hasHandlers():
            logger.handlers.clear()

        if self.config['logging']['save_logs']:
            log_file_path = os.path.join(self.metrics.history_path, 'logging_out.log')
            # Clear the log file if it exists
            if os.path.exists(log_file_path):
                with open(log_file_path, 'w'):
                    pass
            # File handler
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(
                logging.Formatter('[%(levelname)s][%(asctime)s]: %(message)s', datefmt='%H:%M:%S'))
            logger.addHandler(file_handler)

        if self.config['logging']['print_logs']:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(
                logging.Formatter('[%(levelname)s][%(asctime)s]: %(message)s', datefmt='%H:%M:%S'))
            logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

    def getParticipatingClientLoss(self, participating_clients):
        total_loss = {task: 0.0 for task in self.tasks}
        total_samples = 0
        # Loop through each participating client
        for client in participating_clients:
            # Determine the loader to use
            loader = client.dataloader  # Assuming each client has their own dataloader attribute
            # Determine the model to use
            model = self.model_cuda if self.boost_w_gpu else self.model
            for m in model:
                model[m].eval()
            with torch.no_grad():
                loss_fn = self.experiment_module.get_loss()
                for batch in loader:
                    images = batch[0].to(device)  # if device != self.config['data']['trainset_device'] else batch[0]
                    labels = {task: batch[idx + 1].to(device) for idx, task in enumerate(
                        self.tasks)}  # if device != self.config['data']['trainset_device'] else batch[idx + 1] for idx, task in enumerate(self.tasks)}
                    rep, mask = model['rep'](images, None)
                    for task in self.tasks:
                        out, _ = model[task](rep, None)
                        loss = loss_fn[task](out, labels[task]).item()
                        total_loss[task] += loss * images.size(0)
                    total_samples += images.size(0)
            for m in model:
                model[m].train()
        # Calculate average loss for each task
        average_loss = {task: total_loss[task] / total_samples for task in self.tasks}
        average_loss_np = np.array([average_loss[task] for task in self.tasks])
        return average_loss_np

    def estimateG_T_G(self, matrices, method):
        d, m = matrices[0].shape
        mem_dev = self.config["model_device"]
        comp_dev = device
        with torch.no_grad():
            if 'randsvd' in method:
                eigNb = int(d / (2 * np.sqrt(d * m)) * self.config['proposed_approx_extra_upload_d'])
                if 'direct' in method:
                    direct_recon = torch.zeros(d, m, device=comp_dev)
                    for Ai in matrices:
                        rAi = rearrange_matrix(Ai.to(comp_dev))
                        U, S, V = randomized_svd(A=rAi, n_components=eigNb, n_oversamples=eigNb // 10, niter=2)
                        direct_recon += reverse_rearrange_matrix(
                            (U.to(comp_dev) @ torch.diag(S).to(comp_dev) @ V.to(comp_dev).T).to(comp_dev), *Ai.shape)
                    B_T_B_approx = direct_recon.T @ direct_recon
                    # myDel(rAi, U, S, V, direct_recon)
                    return B_T_B_approx.cpu().numpy()
                else:
                    Us = [];
                    Ss = [];
                    Vs = []
                    for Ai in matrices:
                        rAi = rearrange_matrix(Ai.to(comp_dev))
                        U, S, V = randomized_svd(A=rAi, n_components=eigNb, n_oversamples=eigNb // 10, niter=2)
                        Us.append(U.to(mem_dev));
                        Ss.append(S.to(mem_dev)), Vs.append(V.to(mem_dev))
                    # myDel(rAi, U, S, V)
                    C_list = [reverse_rearrange_matrix(
                        Us[i].to(comp_dev) @ torch.diag(Ss[i].to(comp_dev)) @ Vs[i].to(comp_dev).T, d, m).to(mem_dev)
                              for i in range(len(matrices))]
                    D_list = [(matrices[i].to(comp_dev) - C_list[i].to(comp_dev)).to(mem_dev) for i in
                              range(len(matrices))]
                    rsumC_full = rearrange_matrix(sum(C_list).to(comp_dev))
                    U, S, V = randomized_svd(A=rsumC_full.to(comp_dev), n_components=eigNb, n_oversamples=eigNb // 10,
                                             niter=2)
                    sumC = reverse_rearrange_matrix(U @ torch.diag(S) @ V.T, d, m)
                    # myDel(rsumC_full)
                    term1 = sum(A_i.to(comp_dev).T @ A_i.to(comp_dev) for A_i in matrices)
                    term2 = sum(
                        C_list[i].to(comp_dev).T @ (sumC - C_list[i].to(comp_dev)) for i in range(len(matrices)))
                    term3 = 2 * sum(
                        D_list[i].to(comp_dev).T @ (sumC - C_list[i].to(comp_dev)) for i in range(len(matrices)))
                    term4 = torch.zeros_like(term3) if \
                        (sum(torch.linalg.norm(C_list[i]) ** 2 for i in range(len(matrices))) / sum(
                            torch.linalg.norm(D_list[i]) ** 2 for i in range(len(matrices)))) ** (0.5) > 1.5 \
                        else (len(matrices) - 1) * sum(D.T.to(comp_dev) @ D.to(comp_dev) for D in D_list)
                    B_T_B_approx = (term1 + term2 + term3 + term4)
                    # myDel(term1, term2, term3, term4,rsumC_full, sumC, C_list, D_list)
                    return B_T_B_approx.cpu().numpy()
            elif 'topk' in method:
                compression_rate = 1 / m * self.config['proposed_approx_extra_upload_d']
                if 'direct' in method:
                    direct_recon = torch.zeros(d, m, device=comp_dev)
                    for Ai in matrices:
                        direct_recon += top_k_compression(Ai.to(comp_dev), compression_rate)
                    B_T_B_approx = direct_recon.T @ direct_recon
                    # myDel(rAi, direct_recon)
                    return B_T_B_approx.cpu().numpy()
                else:
                    C_list = [top_k_compression(matrices[i].to(comp_dev), compression_rate).to(mem_dev) for i in
                              range(len(matrices))]
                    D_list = [(matrices[i].to(comp_dev) - C_list[i].to(comp_dev)).to(mem_dev) for i in
                              range(len(matrices))]
                    rsumC_full = sum(C_list).to(comp_dev)
                    sumC = top_k_compression(rsumC_full, compression_rate)
                    term1 = sum(A_i.to(comp_dev).T @ A_i.to(comp_dev) for A_i in matrices)
                    term2 = sum(
                        C_list[i].to(comp_dev).T @ (sumC - C_list[i].to(comp_dev)) for i in range(len(matrices)))
                    term3 = 2 * sum(
                        D_list[i].to(comp_dev).T @ (sumC - C_list[i].to(comp_dev)) for i in range(len(matrices)))
                    term4 = torch.zeros_like(term3) if \
                        (sum(torch.linalg.norm(C_list[i]) ** 2 for i in range(len(matrices))) / sum(
                            torch.linalg.norm(D_list[i]) ** 2 for i in range(len(matrices)))) ** (0.5) > 1.5 \
                        else (len(matrices) - 1) * sum(D.T.to(comp_dev) @ D.to(comp_dev) for D in D_list)
                    B_T_B_approx = (term1 + term2 + term3 + term4)
                    # myDel(term1, term2, term3, term4,rsumC_full, sumC, C_list, D_list)
                    return B_T_B_approx.cpu().numpy()
            logging.info("!! Error: Unknown method for approximation of covariance of Jacobian matrix !!")

    def c_aggregate(self, update, c_clone):
        c_delta_list = list(update.values())
        # c_delta_cache = list(zip(c_delta_list))
        c_global = c_clone
        beta = self.config['algorithm_args'][self.config['algorithm']]['beta']
        # update global model
        avg_weight = torch.tensor(
            [
                1 / self.config["nb_of_participating_clients"]
                for _ in range(self.config["nb_of_participating_clients"])
            ],
            device=device,
        )

        # update global control
        for c_g, c_c, c_del, g_global in zip(c_global, c_clone, zip(*c_delta_list), self.g_global):
            c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1)
            c_g.data += (
                                self.config["nb_of_participating_clients"] / self.config['clients']['total']
                        ) * c_del

            c_c.data += c_del
            g_global.data += beta * c_c.data + (1 - beta) * g_global.data

    def g_aggregate(self, update):
        # c_global = update['c_global']
        c_delta_list = list(update.values())
        beta = self.config['algorithm_args'][self.config['algorithm']]['beta']
        # update global model
        avg_weight = torch.tensor(
            [
                1 / self.config["nb_of_participating_clients"]
                for _ in range(self.config["nb_of_participating_clients"])
            ],
            device=device,
        )

        # update global control
        for c_g, c_del, g_global in zip(self.c_global, zip(*c_delta_list), self.g_global):
            c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1)
            c_g.data += c_del

            g_global.data += beta * c_g.data + (1 - beta) * g_global.data

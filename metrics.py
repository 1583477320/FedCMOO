import os
import json
import torch
import numpy as np
from datetime import datetime

class Metrics:
    def __init__(self, config, tasks):
        self.config = config

        # Here, we define which metrics are considered for each task
        # It should be dictionary, however, if list, then it means all tasks have the same eval metrics
        if self.config['experiment'] in ['MultiMNIST', "CIFAR10_MNIST", "MNIST_FMNIST", 'CelebA', 'CelebA_CNN', 'CelebA5', 'CelebA5_CNN']:
            self.eval_metrics = ['accuracy', 'loss']
        elif self.config['experiment'] in ['QM9']:
            self.eval_metrics = ['loss']

        if isinstance(self.eval_metrics, list):
            temp = self.eval_metrics.copy()
            self.eval_metrics = {t:temp for t in tasks}

        self.tasks = tasks
        if self.config['reload_exp']['flag'] is True:
            starting_round = self.reload_exp()
            self.current_round = starting_round + 1
            with open(os.path.join(self.history_path, 'metrics_out.json'), 'r') as f:
                old_metrics_dict = json.load(f)
            self.train_metrics = old_metrics_dict['train']
            self.val_metrics = old_metrics_dict['val']
            self.test_metrics = old_metrics_dict['test']
        else:
            self.current_round = 0
            self.train_metrics = {task: {temp :{} for temp in self.eval_metrics[task]} for task in tasks}
            self.val_metrics = {task: {temp :{} for temp in self.eval_metrics[task]} for task in tasks}
            self.test_metrics = {task: {temp :{} for temp in self.eval_metrics[task]} for task in tasks}
            self.history_path, self.exp_id = self._create_experiment_history_folder()
            with open(os.path.join(self.history_path, 'exp_config.json'), 'w') as f:
                json.dump(self.config, f, indent=4)
            self.starting_round = 0

    def _create_experiment_history_folder(self):
        base_path = os.path.join(os.getcwd(), self.config['paths']['experiment_history'])
        os.makedirs(base_path, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_id =  self.config["experiment"]+'_'+str(timestamp) if self.config['exp_identifier'] is None or not bool(self.config['exp_identifier'].split()) else self.config['exp_identifier'] 
        exp_path =  os.path.join(base_path, f'{exp_id}')
        os.makedirs(exp_path, exist_ok=True)
        return exp_path, exp_id

    def save_model(self, model):
        round_num = self.current_round
        for m in model.keys():
            original_device = next(next(iter(model.values())).parameters()).device
            model[m].to('cpu')
            model_path = os.path.join(self.history_path, f'latest_model_{m}.pth')
            torch.save(model[m].state_dict(), model_path)
            model[m].to(original_device)
        with open(os.path.join(self.history_path, 'round.txt'), 'w') as f:
            f.write(str(round_num))

    def load_model(self, model):
        for m in model.keys():
            original_device = next(next(iter(model.values())).parameters()).device
            model[m].to('cpu')
            model_path = os.path.join(self.history_path, f'latest_model_{m}.pth')
            model[m].load_state_dict(torch.load(model_path))
            model[m].to(original_device)
    
    def reload_exp(self):
        base_path = os.path.join(os.getcwd(), self.config['paths']['experiment_history'])
        self.history_path = os.path.join(base_path, self.config['reload_exp']['folder_name'])
        self.exp_id = self.config['reload_exp']['folder_name']
        round_file = os.path.join(self.history_path, 'round.txt')
        self.current_round = int(f.read().strip())
        return self.current_round

    def update_metrics(self, phase, updates):
        for task in self.tasks:
            for metric in self.eval_metrics[task]: 
                if phase == 'train':
                    self.train_metrics[task][metric][self.current_round] = updates[task][metric] 
                if phase == 'val':
                    self.val_metrics[task][metric][self.current_round] = updates[task][metric] 
                if phase == 'test':
                    self.test_metrics[task][metric][self.current_round] = updates[task][metric]
        self.log_metrics()

    def log_metrics(self):
        round_num = self.current_round
        metrics = {
            'train': self.train_metrics,
            'val': self.val_metrics,
            'test': self.test_metrics
        }
        with open(os.path.join(self.history_path, 'metrics_out.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

# Define metrics initialization based on the experiment
def get_metrics(config, exp_module):
    return Metrics(config, tasks=exp_module.get_tasks())

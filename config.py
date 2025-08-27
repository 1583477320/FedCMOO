import json
from collections import defaultdict
from torch.cuda import is_available as cuda_check
import logging

logging.basicConfig(level=logging.INFO)
import numpy as np


class Config(object):
    """Configuration module."""

    def __init__(self, config_path):
        if isinstance(config_path, str):
            self.path = config_path
            # Load config file
            with open(config_path, 'r') as config_file:
                config_dict = json.load(config_file)
                self.config = self._convert_to_defaultdict(config_dict)
        else:
            self.config = self._convert_to_defaultdict(config_path)

        # Initialize default values
        self.initialize_defaults()

        # Check configuration
        self.check()

    def _convert_to_defaultdict(self, d):
        """Recursively converts a dictionary to a defaultdict with default values None."""
        if not isinstance(d, dict):
            return d
        return defaultdict(lambda: None, {k: self._convert_to_defaultdict(v) for k, v in d.items()})

    def initialize_defaults(self):
        """Initialize default values for specific fields and subfields."""
        defaults = {
            'clients': {
                'total': 100,
                'do_test': False,
                'test_partition': 0,
                'data_points': 200,
                'partition_seed': 10
            },
            'algorithm': 'fsmgda',
            'algorithm_args': {'fsmgda': {'scale_decoders': True,
                                          'count_decoders': False,
                                          'normalize_updates': False,
                                          'normalize_local_iters': False,
                                          "compression": False},
                               'fedcmoo': {'normalize_updates': False,
                                           'count_decoders': False,
                                           'scale_decoders': True,
                                           'scale_lr': None,
                                           'scale_momentum': None,
                                           'scale_n_iter': 1},
                               'fedcmoo_pref': {'normalize_updates': False,
                                                'count_decoders': False,
                                                'scale_decoders': True,
                                                'preference': 'uniform',
                                                'min_weight_multiplier': 0.2},
                               "fedadam": {"scale_decoders": False,
                                           "count_decoders": False,
                                           "normalize_updates": False,
                                           "scale_lr": 0.001,
                                           "scale_momentum": 0,
                                           "scale_n_iter": 1000,
                                           "control_momentum": 1,
                                           "beta": 1},
                               "fsmgda_vr": {"scale_decoders": True,
                                             "count_decoders": False,
                                             "normalize_updates": False,
                                             "compression": False,
                                             "beta": 0.99,
                                             "lipschitz":0.1}
                               },
            "experiment": "MultiMNIST",
            "exp_identifier": None,
            'nb_of_participating_clients': 20,
            'max_round': 50,
            'wandb': {
                'flag': False,
                'wandb_runname': '',
                'wandb_project_name': 'default_project',
                'run_group': ''
            },
            'paths': {
                'data': './data',
                'experiments': './experiments',
                'experiment_history': './experiment_history'
            },
            'model_device': 'cuda',
            'data_seed': 1,
            'hyperparameters': {
                'global_lr': 1,
                'local_training': {
                    'optimizer': 'SGD',
                    'batch_size': 128,
                    'nb_of_local_rounds': 10,
                    'local_lr': 0.3,
                    'local_momentum': 0,
                    "local_lr_scheduler_flag": False
                }
            },
            "proposed_approx_extra_upload_d": 1,
            "proposed_approx_method": "randsvd",
            'data': {
                'distribution': 'dirichlet_first_label',  # uniform, dirichlet_first_label, dirichlet_all_labels
                'test_batch_size': 400,
                'diric_alpha': 0.3,
                'pre_transform': True,
                'testset_device': 'cuda',
                'trainset_device': 'cuda',
                'valset_device': 'cuda',
                'val_seed': 1,
                'val_ratio': 0
            },
            'metrics': {'train_period': 1,
                        'test_period': 3,
                        'val_period': 0,
                        'model_save_period': 10},
            'reload_exp': {'flag': False,
                           'folder_name': 'folder_name'
                           },
            "logging": {"save_logs": True,
                        "print_logs": True}
        }

        self.config = self._apply_defaults(self.config, defaults)

    def _apply_defaults(self, config, defaults):
        """Recursively apply default values to the config."""
        for key, value in defaults.items():
            if isinstance(value, dict):
                if key not in config or config[key] is None:
                    config[key] = defaultdict(lambda: None)
                config[key] = self._apply_defaults(config[key], value)
            elif config[key] is None:
                config[key] = value
        return config

    def check(self):
        """Check the configuration for required fields and constraints."""
        "Device check if cuda is available"
        if not cuda_check():
            tempDeviceCheck = []
            if self.config['model_device'] == 'cuda':
                tempDeviceCheck.append('model_device')
                self.config['model_device'] = 'cpu'
            for temp in ['train', 'val', 'test']:
                if self.config['data'][f'{temp}set_device'] == 'cuda':
                    tempDeviceCheck.append(f'{temp}set_device')
                    self.config['data'][f'{temp}set_device'] = 'cpu'
            if tempDeviceCheck:
                logging.info('You selected cuda for ' + " - ".join(
                    tempDeviceCheck) + ', but cuda is not available on your machine. They are changed to cpu!')
        else:
            for temp in ['train', 'val', 'test']:
                if self.config['data'][f'{temp}set_device'] == 'cuda' and self.config['data']['pre_transform'] is False:
                    logging.info(
                        'You selected cuda for ' + temp + 'set and pre_transform False. pre_transform must be True with any cuda. So, it is changed to True.')
                    self.config['data']['pre_transform'] = True

        # Default seeds
        if self.config['data']['val_seed'] is None:
            self.config['data']['val_seed'] = 1
        if self.config['clients']['partition_seed'] is None:
            self.config['clients']['partition_seed'] = int(np.random.randint(1, 6464))

        if self.config['wandb']['flag']:
            if not self.config['wandb']['wandb_runname'].split():
                self.config['wandb']['wandb_runname'] = self.config['exp_identifier']
            elif not self.config['exp_identifier'].split():
                self.config['exp_identifier'] = self.config['wandb']['wandb_runname']

        # Default experiment history path

        if self.config['paths']['experiment_history'] is None or not bool(
                self.config['paths']['experiment_history'].split()):
            self.config['paths']['experiment_history'] = "./experiment_history"


def base_config_set(base_config_file_path, experiment, algorithm):
    # Read the config file
    # with open(base_config_file_path, 'r') as file:
    #     d = json.load(file)
    d = Config(base_config_file_path).config

    # Modify the config
    d["algorithm"] = algorithm
    d["experiment"] = experiment

    # Update data points based on the experiment
    if experiment == "CelebA" or experiment == "CelebA_CNN" or experiment == "CelebA5" or experiment == "CelebA5_CNN":
        d["clients"]["data_points"] = int(162770 / d["clients"]["total"])
    elif experiment == "CIFAR10_MNIST":
        d["clients"]["data_points"] = int(50000 / d["clients"]["total"])
    elif experiment == "MultiMNIST" or experiment == "MNIST_FMNIST":
        d["clients"]["data_points"] = int(60000 / d["clients"]["total"])
    elif experiment == "QM9":
        d["clients"]["total"] = 20
        d["nb_of_participating_clients"] = 4
        d["clients"]["data_points"] = int(40_000 / d["clients"]["total"])

    if experiment == "CelebA":
        d["data"]['distribution'] = "dirichlet_first_label"
    elif experiment == "CelebA5":
        d["data"]['distribution'] = "dirichlet_first_label"

    # Change model device to CPU for certain conditions
    if (algorithm == "fedcmoo" or algorithm == "fedcmoo_pref") and experiment == "CelebA":
        d["model_device"] = "cpu"

    if experiment == "CelebA":
        d["hyperparameters"]["global_lr"] = 1
        d["hyperparameters"]["local_training"]["local_lr"] = 0.2
    elif experiment == "CelebA5":
        d["hyperparameters"]["global_lr"] = 1.6
        d["hyperparameters"]["local_training"]["local_lr"] = 0.3

    elif experiment == "MultiMNIST":
        if algorithm == "fsmgda":
            d["hyperparameters"]["global_lr"] = 2.0
            d["hyperparameters"]["local_training"]["local_lr"] = 0.1
        elif algorithm == "fedcmoo":
            d["hyperparameters"]["global_lr"] = 1.2
            d["hyperparameters"]["local_training"]["local_lr"] = 0.3
        elif algorithm == "fedcmoo_pref":
            d["hyperparameters"]["global_lr"] = 1.6
            d["hyperparameters"]["local_training"]["local_lr"] = 0.3
        elif algorithm == "fedadam":
            d["hyperparameters"]["global_lr"] = 0.005
            d["hyperparameters"]["local_training"]["local_lr"] = 0.002
        elif algorithm == "fsmgda_vr":
            d["hyperparameters"]["global_lr"] = 1.6
            d["hyperparameters"]["local_training"]["local_lr"] = 0.05
            d["hyperparameters"]["local_training"]["local_lr_scheduler_flag"] = False

    elif experiment == "MNIST_FMNIST":
        if algorithm == "fsmgda":
            d["hyperparameters"]["global_lr"] = 2
            d["hyperparameters"]["local_training"]["local_lr"] = 0.1
        elif algorithm == "fedcmoo":
            d["hyperparameters"]["global_lr"] = 1.2
            d["hyperparameters"]["local_training"]["local_lr"] = 0.5
        elif algorithm == "fedcmoo_pref":
            d["hyperparameters"]["global_lr"] = 1.6
            d["hyperparameters"]["local_training"]["local_lr"] = 0.3
        elif algorithm == "fedcmoo_test":
            d["hyperparameters"]["global_lr"] = 1.2
            d["hyperparameters"]["local_training"]["local_lr"] = 0.1
        
    elif experiment == "CIFAR10_MNIST":
        d["hyperparameters"]["global_lr"] = 0.1
        d["hyperparameters"]["local_training"]["local_lr"] = 0.05

    elif experiment == "QM9":
        d['hyperparameters']['local_training']['nb_of_local_rounds'] = 14
        d["hyperparameters"]["global_lr"] = 1
        d["hyperparameters"]["local_training"]["optimizer"] = 'Adam'
        d["hyperparameters"]["local_training"]["local_lr"] = 0.01

    return d


# Example usage
if __name__ == "__main__":
    config = Config("path_to_your_config_file.json")
    print(dict(config.config))  # Convert defaultdict to dict for readability

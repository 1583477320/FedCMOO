import torch
import math
from torch.optim.optimizer import Optimizer
from collections import defaultdict
from config import Config
import logging


class StormOptimizer(Optimizer):
    """PyTorch implementation of STORM (STOchastic Recursive Momentum) optimizer"""

    def __init__(self, params, lr=1.0, g_max=0.01, momentum=100.0, eta=10.0):
        """
        Args:
            params (iterable): iterable of parameters to optimize
            lr (float): learning rate scaling (k in paper)
            g_max (float): initial value for gradient squared accumulator
            momentum (float): momentum scaling factor
            eta (float): initial denominator for adaptive learning rate (w in paper)
        """
        defaults = dict(lr=lr, g_max=g_max, momentum=momentum, eta=eta)
        super(StormOptimizer, self).__init__(params, defaults)

        # Initialize state for each parameter
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['previous_iterate'] = p.data.clone()
                state['sum_grad_squared'] = torch.full_like(p, group['g_max'] ** 3)
                state['grad_estimate'] = torch.zeros_like(p)
                state['maximum_gradient'] = torch.full_like(p, group['g_max'])
                state['sum_estimates_squared'] = torch.full_like(p, 0.01)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Store current parameters and gradients
        current_params = {}
        current_grads = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                current_params[p] = p.data.clone()
                current_grads[p] = p.grad.data.clone()

        # Compute gradients at previous iterate
        with torch.enable_grad():
            # Set parameters to previous values
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    p.data.copy_(state['previous_iterate'])

            # Recompute loss and gradients at previous iterate
            if closure is not None:
                loss_prev = closure()

            # Collect gradients at previous iterate
            grads_at_prev = {}
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grads_at_prev[p] = p.grad.data.clone()

        # Restore current parameters and gradients
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data.copy_(current_params[p])
                p.grad.data.copy_(current_grads[p])

        # Update parameters and state
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            eta = group['eta']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # Update maximum gradient
                grad_norm = torch.norm(grad)
                state['maximum_gradient'] = torch.maximum(
                    state['maximum_gradient'],
                    torch.full_like(state['maximum_gradient'], grad_norm))

                # Update sum of squared gradients
                state['sum_grad_squared'] += grad ** 2

                # Compute adaptive learning rate
                adaptive_eta = lr * (eta + state['sum_grad_squared']) ** (-1 / 3)

                # Compute momentum coefficient
                beta = torch.minimum(
                    torch.tensor(1.0, device=p.device),
                    momentum * adaptive_eta ** 2)

                # Update gradient estimate (variance reduction)
                grad_estimate = grad + (1 - beta) * (
                        state['grad_estimate'] - grads_at_prev[p])

                # Clip gradient estimate
                grad_estimate = torch.clamp(
                    grad_estimate,
                    -state['maximum_gradient'],
                    state['maximum_gradient'])

                # Update state
                state['grad_estimate'] = grad_estimate
                state['sum_estimates_squared'] += grad_estimate ** 2
                state['previous_iterate'] = p.data.clone()

                # Update parameters
                p.data.add_(-adaptive_eta * grad_estimate)

        return loss

def lr_scheduler(config, round_num):
    if config["experiment"] == 'MultiMNIST':  # LR scheduler
        if round_num % 22 == 0 and round_num != 0 and round_num < 31:
            # Halve the learning rate
            new_lr = config["hyperparameters"]["local_training"]["local_lr"] * 0.10
            config["hyperparameters"]["local_training"]["local_lr"] = new_lr
            logging.info(f"Round {round_num}: Adjusting learning rate to {new_lr:.6f}")
        elif round_num % 30 == 0 and 21 <= round_num < 51:
            # Halve the learning rate
            new_lr = config["hyperparameters"]["local_training"]["local_lr"] * 2.5
            config["hyperparameters"]["local_training"]["local_lr"] = new_lr
            # config["algorithm_args"]["fsmgda_vr"]["beta"] = 0.97
            logging.info(f"Round {round_num}: Adjusting learning rate to {new_lr:.6f}")
        elif round_num % 30 == 0 and 51 <= round_num < 101:
            # Halve the learning rate
            new_lr = config["hyperparameters"]["local_training"]["local_lr"] * 0.559
            config["hyperparameters"]["local_training"]["local_lr"] = new_lr
            logging.info(f"Round {round_num}: Adjusting learning rate to {new_lr:.6f}")
        elif round_num % 30 == 0 and 101 <= round_num < 151:
            # Halve the learning rate
            new_lr = config["hyperparameters"]["local_training"]["local_lr"] * 0.609
            config["hyperparameters"]["local_training"]["local_lr"] = new_lr
            logging.info(f"Round {round_num}: Adjusting learning rate to {new_lr:.6f}")
        elif round_num % 30 == 0 and 151 <= round_num < 201:
            # Halve the learning rate
            new_lr = config["hyperparameters"]["local_training"]["local_lr"] * 0.651
            config["hyperparameters"]["local_training"]["local_lr"] = new_lr
            logging.info(f"Round {round_num}: Adjusting learning rate to {new_lr:.6f}")
        elif round_num % 30 == 0 and 201 <= round_num < 251:
            # Halve the learning rate
            # new_lr = config["hyperparameters"]["local_training"]["local_lr"] * 1.051
            new_lr = 0.008
            config["hyperparameters"]["local_training"]["local_lr"] = new_lr
            logging.info(f"Round {round_num}: Adjusting learning rate to {new_lr:.6f}")
            
    elif config["experiment"] == 'MNIST_FMNIST':  # LR scheduler
        if round_num % 22 == 0 and round_num != 0 and round_num < 31:
            # Halve the learning rate
            new_lr = config["hyperparameters"]["local_training"]["local_lr"] * 0.15
            config["hyperparameters"]["local_training"]["local_lr"] = new_lr
            logging.info(f"Round {round_num}: Adjusting learning rate to {new_lr:.6f}")
        elif round_num % 30 == 0 and 21 <= round_num < 51:
            # Halve the learning rate
            new_lr = config["hyperparameters"]["local_training"]["local_lr"] * 0.8
            config["hyperparameters"]["local_training"]["local_lr"] = new_lr
            # config["algorithm_args"]["fsmgda_vr"]["beta"] = 0.97
            logging.info(f"Round {round_num}: Adjusting learning rate to {new_lr:.6f}")
        elif round_num % 30 == 0 and 51 <= round_num < 101:
            # Halve the learning rate
            new_lr = config["hyperparameters"]["local_training"]["local_lr"] * 0.509
            config["hyperparameters"]["local_training"]["local_lr"] = new_lr
            logging.info(f"Round {round_num}: Adjusting learning rate to {new_lr:.6f}")
        elif round_num % 30 == 0 and 101 <= round_num < 151:
            # Halve the learning rate
            new_lr = config["hyperparameters"]["local_training"]["local_lr"] * 0.609
            config["hyperparameters"]["local_training"]["local_lr"] = new_lr
            logging.info(f"Round {round_num}: Adjusting learning rate to {new_lr:.6f}")
        elif round_num % 30 == 0 and 151 <= round_num < 201:
            # Halve the learning rate
            new_lr = config["hyperparameters"]["local_training"]["local_lr"] * 0.651
            config["hyperparameters"]["local_training"]["local_lr"] = new_lr
            logging.info(f"Round {round_num}: Adjusting learning rate to {new_lr:.6f}")
        elif round_num % 30 == 0 and 201 <= round_num < 251:
            # Halve the learning rate
            # new_lr = config["hyperparameters"]["local_training"]["local_lr"] * 1.051
            new_lr = 0.03
            config["hyperparameters"]["local_training"]["local_lr"] = new_lr
            logging.info(f"Round {round_num}: Adjusting learning rate to {new_lr:.6f}")

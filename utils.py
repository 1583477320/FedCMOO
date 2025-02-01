import torch
import copy
import numpy as np
import logging

def reset_gradients(model):
    for param in model.parameters():
        param.grad = None

def model_to_dict(model, **kwargs):
    """Convert a model's parameters to a dictionary."""
    state_dict = model.state_dict()
    if 'device' in kwargs.keys():
        return {name: param.clone().to(kwargs['device']) for name, param in state_dict.items()}
    return {name: param.clone() for name, param in state_dict.items()}

def dict_to_model(model, param_dict):
    """Load a dictionary of parameters into a model."""
    state_dict = model.state_dict()
    for name, param in param_dict.items():
        if name in state_dict:
            state_dict[name].copy_(param)
    model.load_state_dict(state_dict)
    return model

def transfer_parameters(source_model_dict, target_model_dict):
    """
    Transfer parameters between models in two dictionaries.
    
    Parameters:
    source_model_dict (dict): Dictionary containing the source models.
    target_model_dict (dict): Dictionary containing the target models.
    """
    for key in source_model_dict:
        if key in target_model_dict:
            # Determine the device of the source and target models
            source_model = source_model_dict[key]
            target_model = target_model_dict[key]
            
            source_device = next(source_model.parameters()).device
            target_device = next(target_model.parameters()).device

            # Transfer the state_dict from source_model to target_model
            source_state_dict = source_model.state_dict()
            target_state_dict = {k: v.to(target_device) for k, v in source_state_dict.items()}
            
            target_model.load_state_dict(target_state_dict)
        else:
            logging.warning(f"Warning: Key '{key}' not found in target_model_dict")

def update_average(averaged_updates, new_updates, tasks, weight):
    for task in tasks:
        if new_updates[task]['rep'] is not None and new_updates[task][task] is not None:
            for key in new_updates[task]['rep']:
                averaged_updates[task]['rep'][key] += new_updates[task]['rep'][key] * weight
            for key in new_updates[task][task]:
                averaged_updates[task][task][key] += new_updates[task][task][key] * weight
    return averaged_updates
    
def average_updates(all_updates, tasks, algo):
    if algo in ['fsmgda']:
        """Average updates for each task across all clients, accounting for tasks not calculated by all clients."""
        averaged_updates = {task: {'rep': {}, task: {}} for task in tasks}
        task_client_counts = {task: 0 for task in tasks}
    
        # Initialize the averaged updates dictionary
        for task in tasks:
            for key in all_updates[0][task]['rep']:
                averaged_updates[task]['rep'][key] = torch.zeros_like(all_updates[0][task]['rep'][key])
            for key in all_updates[0][task][task]:
                averaged_updates[task][task][key] = torch.zeros_like(all_updates[0][task][task][key])
    
        # Sum the updates from all clients and count clients per task
        for updates in all_updates:
            for task in tasks:
                if updates[task]['rep'] is not None and updates[task][task] is not None:
                    task_client_counts[task] += 1
                    for key in updates[task]['rep']:
                        averaged_updates[task]['rep'][key] += updates[task]['rep'][key]
                    for key in updates[task][task]:
                        averaged_updates[task][task][key] += updates[task][task][key]
    
        # Divide by the number of clients that provided updates for each task to get the average
        for task in tasks:
            if task_client_counts[task] > 0:
                for key in averaged_updates[task]['rep']:
                    averaged_updates[task]['rep'][key] /= task_client_counts[task]
                for key in averaged_updates[task][task]:
                    averaged_updates[task][task][key] /= task_client_counts[task]
    if algo in ['fedcmoo','fedcmoo_pref']:
        averaged_updates = {'rep': {}, **{task: {} for task in tasks}}
        client_count = len(all_updates)

        # Initialize the averaged updates dictionary
        for key in all_updates[0]['rep']:
            averaged_updates['rep'][key] = torch.zeros_like(all_updates[0]['rep'][key])
        for task in tasks:
            for key in all_updates[0][task]:
                averaged_updates[task][key] = torch.zeros_like(all_updates[0][task][key])

        # Sum the updates from all clients
        for updates in all_updates:
            for key in updates['rep']:
                averaged_updates['rep'][key] += updates['rep'][key]
            for task in tasks:
                for key in updates[task]:
                    averaged_updates[task][key] += updates[task][key]

        # Divide by the number of clients to get the average
        for key in averaged_updates['rep']:
            averaged_updates['rep'][key] /= client_count
        for task in tasks:
            for key in averaged_updates[task]:
                averaged_updates[task][key] /= client_count
    return averaged_updates

def print_gpu_utilization():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the current device
        device = torch.device('cuda')
        
        # Get the current memory usage on the device
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        # Calculate free memory
        free_memory = total_memory - reserved_memory
        
        # Convert bytes to megabytes for readability
        allocated_memory_MB = allocated_memory / (1024 ** 2)
        reserved_memory_MB = reserved_memory / (1024 ** 2)
        free_memory_MB = free_memory / (1024 ** 2)
        total_memory_MB = total_memory / (1024 ** 2)
    
        # print(f"Total GPU memory: {total_memory_MB:.2f} MB")
        # print(f"Allocated GPU memory: {allocated_memory_MB:.2f} MB")
        # print(f"Reserved GPU memory: {reserved_memory_MB:.2f} MB")
        print(f"Free GPU memory: {free_memory_MB:.2f} MB")
    else:
        print("CUDA is not available on this system.")

def return_free_gpu_memory():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the current device
        device = torch.device('cuda')
        
        # Get the current memory usage on the device
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        # Calculate free memory
        free_memory = total_memory - reserved_memory
        free_memory_MB = free_memory / (1024 ** 2)
        return free_memory_MB
    else:
        logging.warning("CUDA is not available on this system.")


def normalize_updates(updates, tasks, config):
    """Normalize the updates of each task.
    Only L2 normalization is here. """
    normalization_type = 'L2' if config['algorithm_args'][config['algorithm']]['normalize_updates'] else None
    def l2_normalize(vector):
        norm = torch.norm(vector, p=2)
        if norm > 0:
            return vector / norm
        return vector

    def l1_normalize(vector):
        norm = torch.norm(vector, p=1)
        if norm > 0:
            return vector / norm
        return vector
    if normalization_type not in ['L1', 'L2']:
        return updates
    if config['algorithm'] in ['fsmgda']:
        # # Create a deep copy of the updates to ensure the original is not modified
        normalized_updates = dict()
        for task in tasks:
            normalized_updates[task] = {'rep':dict(), task:dict()}
            for key in updates[task]['rep']:
                normalized_updates[task]['rep'][key] = updates[task]['rep'][key].clone()
            for key in updates[task][task]:
                normalized_updates[task][task][key] = updates[task][task][key].clone()
        
        for task in tasks:
            combined_updates = []
    
            # Combine 'rep' and task-specific updates into a single vector
            for key in normalized_updates[task]['rep']:
                combined_updates.append(normalized_updates[task]['rep'][key].view(-1))
            for key in normalized_updates[task][task]:
                combined_updates.append(normalized_updates[task][task][key].view(-1))
    
            combined_vector = torch.cat(combined_updates)
    
            # Normalize the combined vector
            if normalization_type == 'L2':
                normalized_vector = l2_normalize(combined_vector)
            elif normalization_type == 'L1':
                normalized_vector = l1_normalize(combined_vector)
            else:
                normalized_vector = combined_vector
    
            # Split the normalized vector back into 'rep' and task-specific updates
            start = 0
            for key in normalized_updates[task]['rep']:
                end = start + normalized_updates[task]['rep'][key].numel()
                normalized_updates[task]['rep'][key] = normalized_vector[start:end].view_as(normalized_updates[task]['rep'][key])
                start = end
            for key in normalized_updates[task][task]:
                end = start + normalized_updates[task][task][key].numel()
                normalized_updates[task][task][key] = normalized_vector[start:end].view_as(normalized_updates[task][task][key])
                start = end
        return normalized_updates
    elif config['algorithm'] in ['fedcmoo'] or 'fedcmoo_pref' in config['algorithm']:
        combined_updates = []

        # Combine 'rep' and all task-specific updates into a single vector
        for key in updates['rep']:
            combined_updates.append(updates['rep'][key].view(-1))
        for task in tasks:
            for key in updates[task]:
                combined_updates.append(updates[task][key].view(-1))

        combined_vector = torch.cat(combined_updates)

        # Normalize the combined vector
        if normalization_type == 'L2':
            normalized_vector = l2_normalize(combined_vector)
        elif normalization_type == 'L1':
            normalized_vector = l1_normalize(combined_vector)
        else:
            normalized_vector = combined_vector

        # Split the normalized vector back into 'rep' and all task-specific updates
        start = 0
        for key in updates['rep']:
            end = start + updates['rep'][key].numel()
            updates['rep'][key] = normalized_vector[start:end].view_as(updates['rep'][key])
            start = end
        for task in tasks:
            for key in updates[task]:
                end = start + updates[task][key].numel()
                updates[task][key] = normalized_vector[start:end].view_as(updates[task][key])
                start = end
        return updates

# This code is from
# Multi-Task Learning as Multi-Objective Optimization
# Ozan Sener, Vladlen Koltun
# Neural Information Processing Systems (NeurIPS) 2018
# https://github.com/intel-isl/MultiObjectiveOptimization
class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-10

    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1 + v2v2 - 2 * v1v2))
        cost = v2v2 + gamma * (v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\\sum c_i x_i|_2^2 st. \\sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, j)] += torch.dot(
                            vecs[i][k], vecs[j][k]
                        ).item()  # torch.dot(vecs[i][k], vecs[j][k]).dataset[0]
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i, i)] += torch.dot(
                            vecs[i][k], vecs[i][k]
                        ).item()  # torch.dot(vecs[i][k], vecs[i][k]).dataset[0]
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.dot(
                            vecs[j][k], vecs[j][k]
                        ).item()  # torch.dot(vecs[j][k], vecs[j][k]).dataset[0]
                c, d = MinNormSolver._min_norm_element_from2(
                    dps[(i, i)], dps[(i, j)], dps[(j, j)]
                )
                if d < dmin:
                    dmin = d
                    sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \\sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m - 1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    @staticmethod
    def _next_point(cur_val, grad, n):
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / proj_grad[proj_grad < 0]
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0])

        skippers = np.sum(tm1 < 1e-7) + np.sum(tm2 < 1e-7)
        t = 1
        if len(tm1[tm1 > 1e-7]) > 0:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) > 0:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad * t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    @staticmethod
    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \\sum c_i vecs[i] and \\sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * np.dot(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i] * sol_vec[j] * dps[(i, j)]
                    v1v2 += sol_vec[i] * new_point[j] * dps[(i, j)]
                    v2v2 += new_point[i] * new_point[j] * dps[(i, j)]
            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec + (1 - nc) * new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec

    @staticmethod
    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \\ sum c_i vecs[i] and \\sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc * sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec


# Taken from https://github.com/OptMN-Lab/sdmgrad/blob/main/mtrl_files/sdmgrad.py
def proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \\sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    [2] Projection onto the probability simplex: An efficient algorithm with a simple proof, and an application
        Weiran Wang, Miguel Á. Carreira-Perpiñán. arXiv:1309.1541
        https://arxiv.org/pdf/1309.1541.pdf
    [3] https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246#file-simplex_projection-py
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    v = v.astype(np.float64)
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - s) / (rho + 1)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w

import gc
def myDel(*objs):
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure all GPU ops are done
    for obj in objs:
        del obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def myAttrDel(obj_ref, attr_name):
    """
    Deletes the attribute of the passed object and clears GPU cache and Python's garbage collection.
    
    Args:
        obj_ref (object): The object reference containing the attribute.
        attr_name (str): The name of the attribute to delete.
    """
    # Synchronize GPU before deletion if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Delete the attribute from the object (e.g., self.initial_round_gradients)
    if hasattr(obj_ref, attr_name):
        delattr(obj_ref, attr_name)
    
    # Empty CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run garbage collection to free up memory
    gc.collect()

def rotate_batch_affine(batch):
    N, C, H, W = batch.size()
    angles = torch.rand(N, device=batch.device) * 60 - 30  # Random angles between -30 and 30 degrees
    
    # Compute the rotation matrices for each angle
    theta = torch.zeros(N, 2, 3, device=batch.device)
    cos_vals = torch.cos(angles * torch.pi / 180.0)
    sin_vals = torch.sin(angles * torch.pi / 180.0)
    
    theta[:, 0, 0] = cos_vals
    theta[:, 0, 1] = -sin_vals
    theta[:, 1, 0] = sin_vals
    theta[:, 1, 1] = cos_vals

    grid = F.affine_grid(theta, batch.size(), align_corners=False)
    return F.grid_sample(batch, grid, align_corners=False)


def nrmse(true, estimate):
    # Check if inputs are NumPy arrays or PyTorch tensors
    if isinstance(true, np.ndarray) and isinstance(estimate, np.ndarray):
        # For NumPy arrays, use NumPy's linear algebra operations
        return float(np.linalg.norm(true - estimate) / np.linalg.norm(true))
    elif torch.is_tensor(true) and torch.is_tensor(estimate):
        # For PyTorch tensors, use PyTorch's linear algebra operations
        return float(torch.linalg.norm(true - estimate) / torch.linalg.norm(true))
    else:
        raise TypeError("Inputs must be both NumPy arrays or both PyTorch tensors.")
import torch

def rearrange_matrix(matrix):
    # Get the device of the input matrix (CPU or CUDA)
    device = matrix.device
    
    # Get the dimensions of the input matrix
    d, m = matrix.shape
    
    # Calculate d_sq as the floor of sqrt(d*m)
    d_sq = int(torch.floor(torch.sqrt(torch.tensor(d * m, dtype=torch.float32, device=device))))
    
    # Initialize the list to hold the chunks
    chunks = []
    
    # Process the matrix by splitting it into chunks of size d_sq x m
    for i in range(0, d, d_sq):
        chunk = matrix[i:i+d_sq, :]
        # If the chunk is smaller than d_sq, pad with zeros on the same device
        if chunk.shape[0] < d_sq:
            padding = torch.zeros((d_sq - chunk.shape[0], m), device=device)
            chunk = torch.cat((chunk, padding), dim=0)
        chunks.append(chunk)
    
    # Concatenate the chunks horizontally (column-wise)
    result_matrix = torch.cat(chunks, dim=1)
    
    return result_matrix

def reverse_rearrange_matrix(rearranged_matrix, d, m):
    # Get the device of the rearranged matrix
    device = rearranged_matrix.device
    
    # Calculate d_sq as the floor of sqrt(d*m)
    d_sq = int(torch.floor(torch.sqrt(torch.tensor(d * m, dtype=torch.float32, device=device))))
    
    # Calculate the number of chunks
    num_chunks = rearranged_matrix.shape[1] // m
    
    # Initialize a list to hold the original matrix chunks
    original_chunks = []
    
    # Extract each chunk and add it back to the original list
    for i in range(num_chunks):
        chunk = rearranged_matrix[:, i*m:(i+1)*m]
        original_chunks.append(chunk)
    
    # Concatenate the chunks vertically to restore the original matrix
    original_matrix = torch.cat(original_chunks, dim=0)
    
    # Truncate the extra rows if any were added during the padding
    original_matrix = original_matrix[:d, :]

    return original_matrix


from typing import Optional, Tuple
import torch
from torch import Tensor
from torch import _linalg_utils as _utils
from torch.overrides import handle_torch_function, has_torch_function

# Utility function to handle mH operation based on PyTorch version
def hermitian_transpose(tensor):
    if hasattr(tensor, 'mH'):
        # If mH is supported (PyTorch version >= 1.9), use it
        return tensor.mH
    else:
        # For earlier versions, manually perform the conjugate transpose
        return torch.conj(tensor).transpose(-2, -1)

def get_approximate_basis(
    A: Tensor, n_components:int, n_oversamples: Optional[int] = 0, niter: Optional[int] = 2, M: Optional[Tensor] = None
) -> Tensor:
    with torch.no_grad():
        niter = 2 if niter is None else niter
        dtype = _utils.get_floating_dtype(A) if not A.is_complex() else A.dtype
        matmul = _utils.matmul
        q = n_components + n_oversamples
        R = torch.randn(A.shape[-1], q, dtype=dtype, device=A.device)
    
        # The following code could be made faster using torch.geqrf + torch.ormqr
        # but geqrf is not differentiable
        
        X = matmul(A, R)
        if M is not None:
            X = X - matmul(M, R)
        Q = torch.linalg.qr(X).Q
        for i in range(niter):
            X = matmul(hermitian_transpose(A), Q)
            if M is not None:
                X = X - matmul(hermitian_transpose(M), Q)
            Q = torch.linalg.qr(X).Q
            X = matmul(A, Q)
            if M is not None:
                X = X - matmul(M, Q)
            Q = torch.linalg.qr(X).Q
        return Q

def randomized_svd(
    A: Tensor,
    Q: Optional[Tensor] = None,
    n_components: Optional[int] = 6,
    n_oversamples: Optional[int] = 0,
    niter: Optional[int] = 2,
    M: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    with torch.no_grad():
        if not torch.jit.is_scripting():
            tensor_ops = (A, M)
            if not set(map(type, tensor_ops)).issubset(
                (torch.Tensor, type(None))
            ) and has_torch_function(tensor_ops):
                return handle_torch_function(
                    svd_lowrank, tensor_ops, A, Q=Q, n_components=n_components, n_oversamples=n_oversamples, niter=niter, M=M
                )
        return _svd_lowrank(A, Q=Q, n_components=n_components, n_oversamples=n_oversamples, niter=niter, M=M)

def _svd_lowrank(
    A: Tensor,
    Q: Optional[Tensor] = None,
    n_components: Optional[int] = 6,
    n_oversamples: Optional[int] = 0,
    niter: Optional[int] = 2,
    M: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    # Algorithm 5.1 in Halko et al., 2009
    
    n_components = 6 if n_components is None else n_components
    m, n = A.shape[-2:]
    matmul = _utils.matmul
    if M is not None:
        M = M.broadcast_to(A.size())

    # # Assume that A is tall
    # if m < n:
    #     A = hermitian_transpose(A)
    #     if M is not None:
    #         M = hermitian_transpose(M)
    if Q is None:
        Q = get_approximate_basis(A, n_components=n_components, n_oversamples=n_oversamples, niter=niter, M=M)
    B = matmul(hermitian_transpose(Q), A)
    if M is not None:
        B = B - matmul(hermitian_transpose(Q), M)
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    V = hermitian_transpose(Vh)
    U = Q.matmul(U)

    # if m < n:
    #     U, V = V, U

    return U[:,:n_components], S[:n_components], V[:,:n_components]

def calculate_k(d1, d2, compression_rate):
    total_elements = d1 * d2
    log2_d1d2 = np.log2(total_elements)

    def comm_cost1(K):
        return K + K * log2_d1d2 / 16

    def comm_cost2(K):
        return K + total_elements / 16

    # Binary search for K
    low, high = 1, total_elements
    target_communication = np.floor(total_elements * compression_rate)

    while low < high:
        mid = (low + high) // 2
        comm_cost = min(comm_cost1(mid), comm_cost2(mid))

        if np.floor(comm_cost) < target_communication:
            low = mid + 1
        else:
            high = mid

    return low
def top_k_compression(A, compression_rate):
    d1, d2 = A.shape
    K = calculate_k(d1, d2, compression_rate)

    # Flatten the tensor
    flat_A = A.reshape(-1)  # Use reshape instead of view

    # Get the K largest absolute values' indices in A
    _, indices = torch.topk(torch.abs(flat_A), K)

    # Create a compressed matrix with zeros
    compressed_A = torch.zeros_like(A).reshape(-1)  # Use reshape instead of view

    # Copy the top-K elements to the compressed tensor
    compressed_A[indices] = flat_A[indices]

    # Reshape it back to the original shape
    compressed_A = compressed_A.reshape(d1, d2)

    return compressed_A


def random_compression(A, compression_rate):
    d1, d2 = A.shape
    n = d1 * d2

    # Calculate the number of non-zero entries to retain based on compression rate
    num_non_zero = int(n * compression_rate)

    # Create a flat mask initialized to zeros
    mask = torch.zeros(n, device = A.device)

    # Randomly select indices to set to 1 in the mask
    selected_indices = torch.randperm(n)[:num_non_zero]
    mask[selected_indices] = 1

    # Reshape the mask back to the original matrix dimensions
    mask = mask.reshape(d1, d2)

    # Return the element-wise product of A and the mask
    compressed_A = A * mask

    return compressed_A


def top_k_compression_dict(data, compression_rate=1):
    # Iterate over each task in the input dictionary
    for task, model_parts in data.items():
        # Collect all tensors from the task into a single 1D tensor
        all_values = []
        for model_part, weights in model_parts.items():
            for weight_name, tensor in weights.items():
                all_values.append(tensor.reshape(-1))  # Flatten all tensors into 1D
        
        # Concatenate all the task's values into a single tensor
        concatenated_values = torch.cat(all_values)
        
        # Calculate K using your calculate_k function
        d1 = concatenated_values.numel()  # Number of elements in the task dictionary
        K = calculate_k(d1, 1, compression_rate)

        # Apply top-K compression on the concatenated values
        _, topk_indices = torch.topk(torch.abs(concatenated_values), K)

        # Create a compressed version (zeros everywhere except top-K indices)
        compressed_values = torch.zeros_like(concatenated_values)
        compressed_values[topk_indices] = concatenated_values[topk_indices]

        # Re-assign compressed values back to the original dictionary
        index = 0
        for model_part, weights in model_parts.items():
            for weight_name, tensor in weights.items():
                tensor_size = tensor.numel()  # Get the number of elements in the tensor
                compressed_tensor = compressed_values[index:index + tensor_size].reshape(tensor.shape)
                data[task][model_part][weight_name] = compressed_tensor
                index += tensor_size
    
    return data


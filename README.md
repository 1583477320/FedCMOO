# FedCMOO: Federated Communication-Efficient Multi-Objective Optimization

Official implementation of "FedCMOO: Federated Communication-Efficient Multi-Objective Optimization" accepted to *AISTATS 2025*


## Requirements

We recommend creating a new environment. To install the required packages, run the following command:
```
pip install -r requirements.txt
```

All experiments were tested on an NVIDIA H100 GPU in our internal cluster using CUDA. The code also supports CPU execution and has been tested!

## Datasets

All datasets, except CelebA and QM9, will be automatically downloaded the first time you run the experiments. However, downloading the CelebA dataset may be problematic due to Google Drive download quotas. This issue even affects PyTorch’s built-in code. Additionally, the QM9 dataset requires extra packages. Here’s how to resolve these issues:

- To use the CelebA dataset, please download it from Kaggle:

[https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data)

After downloading, place the `celeba.zip` file in the `/data/CelebA` directory.

- To download QM9 dataset, we mainly follow [LibMTL QM9 implementation](https://github.com/median-research-group/LibMTL/tree/main/examples/qm9). Please install PyTorch Geometric as follows. Our code will handle the rest on the first run, provided that all required dependencies are installed.

```shell
pip install torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch_sparse==0.6.10 -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch_geometric==2.2.0
```


## Demo Experiments

Once all dependencies are installed, you can run the demo experiment using the `demo.ipynb` notebook. The notebook includes a demo experiment using the MNIST+FMNIST setting. The code will automatically utilize CUDA if available or fall back to the CPU otherwise.

You can also run any experiment using the following terminal command:
```
python run.py “config_file_path.json”
```

## Config Files

We provide a `base_config_file.json`, along with a configuration dictionary generator method ```base_config_set``` in `config.py`, which automatically selects the tuned hyperparameters.

Below are the experimental parameters and their descriptions.

## Configuration File Instructions

- Below, you can find the experimental parameters and their descriptions:

    ```clients``` := Client control settings  
    ├── ```total``` := The number of clients (default: 100)  
    └── ```data_points``` := The number of data samples per client (defaults: 500 for CIFAR10+MNIST, 600 for MultiMNIST and MNIST+FMNIST, 1600 for CelebA experiments)

    ```algorithm``` := One of ["fedcmoo", "fsmgda", "fedcmoo_pref"]

    ```algorithm_args``` := Algorithm-specific settings  
    ├── ```fsmgda``` := Settings for FSMGDA algorithm  
    │   ├── ```scale_decoders``` := Default true, false to not apply task weights to decoder parts (not suggested)  
    │   ├── ```count_decoders``` := Default false, true to consider decoder parts when finding task weights (not suggested)  
    │   ├── ```normalize_updates``` := Default false, true for L2 normalization of updates  
    ├── ```fedcmoo``` := Settings for FedCMOO algorithm  
    │   ├── ```scale_decoders``` := Default true  
    │   ├── ```count_decoders``` := Default false  
    │   ├── ```normalize_updates``` := Default false  
    │   ├── ```scale_lr``` := Learning rate $\beta$ in FindWeights (default: 0.001)  
    │   ├── ```scale_momentum``` := Default 0  
    │   └── ```scale_n_iter``` := $K$ parameter in FindWeights (default: 1000)  
    └── ```fedcmoo_pref``` := Settings for FedCMOO-Pref algorithm  
        ├── ```scale_decoders``` := Default true  
        ├── ```count_decoders``` := Default false  
        ├── ```normalize_updates``` := Default false  
        ├── ```preference``` := Either a list of preference values, e.g., [1, 3], or "uniform" (default: "uniform")  
        └── ```min_weight_multiplier``` := Specifies 1/5 multiplier used in Practical improvement paragraph in the end of Appendix Preference-based Federated Communication-Efficient MOO section (default: 0.2, and set 0 to not use it). 
        
    ```experiment``` := Experiment type (default: "MultiMNIST"), one of ["MultiMNIST", "MNIST_FMNIST", "CIFAR10_MNIST", "CelebA", "CelebA5", "QM9"]

    ```exp_identifier``` := Experiment run name (default: "v1")

    ```proposed_approx_extra_upload_d``` := The upload ($\times d$) amount required for the proposed communication-efficient ApproxGramJacobian subroutine (default: 1)

    ```proposed_approx_method``` := One of ['randsvd' (two-way), 'topk' (two-way), 'randsvd direct' (one-way), 'topk direct' (one-way)] (default: "randsvd"). Please refer to the Appendix: Details of the ApproxGramJacobian Subroutine.

    ```nb_of_participating_clients``` := The number of participating clients per round (default: 10)

    ```max_round``` := Maximum number of rounds (default: 200)

    ```wandb``` := Settings for WandB integration  
    ├── ```flag``` := Flag to enable WandB (default: false)  
    ├── ```wandb_runname``` := WandB run name (default: "")  
    ├── ```wandb_project_name``` := WandB project name (default: "default_project")  
    └── ```run_group``` := Extra attribute to group runs easily (default: "")

    ```paths``` := File path settings  
    ├── ```data``` := Path for the data folder (default: "./data")  
    ├── ```experiments``` := Path for the experiments folder (default: "./experiments")  
    └── ```experiment_history``` := Path for saving results (default: " ")

    ```model_device``` := Device to store models, either "cuda" or "cpu" (default: "cuda")

    ```data_seed``` := Random seed for data distribution (default: 1)

    ```hyperparameters``` := Training hyperparameters  
    ├── ```global_lr``` := Server learning rate $\eta_s$  
    └── ```local_training``` := Local training settings  
        ├── ```optimizer``` := Optimizer for local training (default: "SGD")  
        ├── ```batch_size``` := Batch size for local training (default: 128)  
        ├── ```nb_of_local_rounds``` := Number of local iterations (default: 10)  
        ├── ```local_lr``` := Client learning rate $\eta_c$ (default: 0.3)  
        ├── ```local_momentum``` := Local momentum (default: 0)  
        └── ```local_lr_scheduler_flag``` := Flag to use a learning rate scheduler (default: false).<br>


    ```data``` := Data settings  
    ├── ```distribution``` := Data distribution method, one of ["dirichlet_all_labels" for MultiMNIST, MNIST_FMNIST, CIFAR10_MNIST, "dirichlet_first_labels" for CelebA and CelebA5, "uniform" for QM9]  
    ├── ```test_batch_size``` := Test batch size (default: 3000)  
    ├── ```diric_alpha``` := Alpha parameter for Dirichlet distribution (default: 0.3)  
    ├── ```pre_transform``` := (default: true)  
    ├── ```testset_device``` := Device for storing the test set (default: "cuda")  
    ├── ```trainset_device``` := Device for storing the training set (default: "cuda")  
    ├── ```valset_device``` := Device for storing the validation set (default: "cuda")  
    ├── ```val_ratio``` := Validation set ratio if there is no default split (default: 0.2)  
    └── ```val_seed``` := Random seed for validation set if there is no default split (default: 1)

    ```metrics``` := Metric evaluation settings  
    ├── ```train_period``` := Period for evaluating training set metrics (default: 1, set 0 for never)  
    ├── ```test_period``` := Period for evaluating test set metrics (default: 3, set 0 for never)  
    ├── ```val_period``` := Period for evaluating validation set metrics (default: 3, set 0 for never)  
    └── ```model_save_period``` := Period to save the latest model weights (default: 0, set 0 for never)

    ```logging``` := Logging settings  
    ├── ```save_logs``` := Flag to save logs (default: true)  
    └── ```print_logs``` := Flag to print logs (default: true)


## Citation

If you find our work useful, please cite it as follows:

```latex
@misc{askin2024fedcmoo,
      title={Federated Communication-Efficient Multi-Objective Optimization}, 
      author={Baris Askin and Pranay Sharma and Gauri Joshi and Carlee Joe-Wong},
      year={2024},
      eprint={2410.16398},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.16398}, 
}
```

For questions/comments, please send an email to [Baris Askin](https://askinb.github.io).
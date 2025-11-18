import sys
import server
import logging
import config

logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, 'INFO'), datefmt='%H:%M:%S')

def main():
    if len(sys.argv):
        # config_path = sys.argv[1]
        config_path = r'/FedCMOOz\base_config.json'
        a = config.Config(config_path)
        s = server.Server(a)
        s.boot()
        t = s.train()
    else:
        print("No config json file provided!")

# def main():
#     exp_configs = [config.base_config_set('base_config.json', experiment = "MultiMNIST", algorithm= 'fedadam'),
#                 # config.base_config_set('base_config.json', experiment = "MultiMNIST", algorithm = 'fedcmoo'),
#               config.base_config_set('base_config.json', experiment = "MultiMNIST", algorithm= 'fsmgda')
#               # config.base_config_set('base_config.json', experiment = "MultiMNIST", algorithm= 'fedcmoo_pref'),
#               ]
#     print(exp_configs[0])


if __name__ == "__main__":
    import numpy as np
    import torch
    import os
    import pandas as pd
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    exp_configs = [
        config.base_config_set("base_config.json", experiment="MultiMNIST", algorithm="fsmgda_vr"),
        config.base_config_set('base_config.json', experiment="MultiMNIST", algorithm='fsmgda'),
        config.base_config_set('base_config.json', experiment="MultiMNIST", algorithm='fedcmoo'),
        # config.base_config_set('base_config.json', experiment="CIFAR10_MNIST", algorithm= 'fsmgda_vr'),
        # config.base_config_set('base_config.json', experiment="CIFAR10_MNIST", algorithm= 'fsmgda_vr')
    ]
    print(exp_configs[0]['algorithm_args'][exp_configs[0]['algorithm']]['beta'])
    print(exp_configs[0]['hyperparameters']['global_lr'])
    print(exp_configs[0]['hyperparameters']['local_training']['local_lr'])
    print(exp_configs[0]['hyperparameters']['local_training']['nb_of_local_rounds'])



    number_of_rounds = 400  # For a quick demo, can be changed
    for i in range(len(exp_configs)):
        exp_configs[i]['max_round'] = number_of_rounds
        exp_configs[i]['swanlab']['flag'] = False

    test_mean_accuracy_list = list()
    test_mean_loss_list = list()

    models = {}

    # mode_list = [5,10,30]
    mode = "algorithm"

    for i, exp_config in enumerate(exp_configs):
        # exp_config['swanlab']['swanlab_runname'] = exp_configs[i]['algorithm'] + "_v43"
        seed = 42
        np.random.seed(42)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model_name = exp_configs[i]['algorithm']
        # exp_configs[i]['nb_of_participating_clients'] = mode_list[i]
        s = server.Server(config.Config(exp_config))
        s.boot()
        s.train()
        models[model_name] = s
        mean_accuracy = 0
        for task in s.tasks:
            mean_accuracy += np.array(list(s.metrics.test_metrics[task]['accuracy'].values()))
        mean_accuracy /= len(s.tasks)
        test_mean_accuracy_list.append(mean_accuracy)
        mean_loss = 0
        for task in s.tasks:
            mean_loss += np.array(list(s.metrics.test_metrics[task]['loss'].values()))
        mean_loss /= len(s.tasks)
        test_mean_loss_list.append(mean_loss)

    # 将结果列表转换为DataFrame
    accuracy_df = pd.DataFrame(test_mean_accuracy_list).T
    accuracy_df.columns = [f"acc_{i + 1}" for i in range(len(test_mean_accuracy_list))]

    loss_df = pd.DataFrame(test_mean_loss_list).T
    loss_df.columns = [f"loss_{i + 1}" for i in range(len(test_mean_loss_list))]

    # 建立文件夹
    path = f"D:/PycharmProjects/FMGDA/FedCMOO/result/{mode}"

    # 判断是否存在
    if not os.path.exists(path):
        os.makedirs(path)  # 若不存在则递归创建
        print(f"已创建文件夹: {path}")
    else:
        print(f"文件夹已存在: {path}")
    # 你可以选择将这些DataFrame保存为文件，例如CSV文件

    accuracy_df.to_csv(f'D:/PycharmProjects/FMGDA/FedCMOO/result/{mode}/{exp_configs[0]["experiment"]}_mean_accuracy.csv', index=False)
    loss_df.to_csv(f'D:/PycharmProjects/FMGDA/FedCMOO/result/{mode}/{exp_configs[0]["experiment"]}_mean_loss.csv', index=False)

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    #
    # # ---------- 1. 读取数据 ----------
    # accuracy_df = pd.read_csv("D:/PycharmProjects/FMGDA/FedCMOO/result/mean_accuracy.csv")
    # loss_df = pd.read_csv("D:/PycharmProjects/FMGDA/FedCMOO/result/mean_loss.csv")
    #
    # test_mean_accuracy_list = np.array([accuracy_df[col].tolist() for col in accuracy_df.columns])
    # test_mean_loss_list = np.array([loss_df[col].tolist() for col in loss_df.columns])
    #
    # title = exp_configs[0]["experiment"]
    # algos = ["5", "10", '30']
    # # algos = ["fsmgda_vr"]
    # colors = ["#000000", "#E69F00", "#009E73"]  # 黑, 橙, 绿Q
    # # colors = ["#009E73"]
    #
    # # -------------------- Plot --------------------
    # sns.set_theme(style="whitegrid")
    # plt.rcParams.update({
    #     "font.size": 12,
    #     "axes.titlesize": 14,
    #     "axes.labelsize": 12,
    #     "legend.fontsize": 11,
    #     "lines.linewidth": 2
    # })
    #
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # round_axis = np.arange(1, len(test_mean_accuracy_list[0]) + 1)
    # # ---------- Accuracy ----------
    # for i, algo in enumerate(algos):
    #     mean = test_mean_accuracy_list[i]
    #     std = mean * 0.015
    #     axes[0].plot(round_axis, mean, label=algo, color=colors[i])
    #     axes[0].fill_between(round_axis, mean - std, mean + std,
    #                          color=colors[i], alpha=0.2)  # 阴影
    # axes[0].set_title(f"{title} Mean Test Accuracy")
    # axes[0].set_xlabel("Rounds")
    # axes[0].set_ylabel("Accuracy")
    # # axes[0].set_ylim([0.4, 0.8])
    # axes[0].grid(True, linestyle="--", alpha=0.5)
    # axes[0].legend(loc='lower right')
    #
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    #
    # # 局部放大图嵌在主图内部右下角
    # axins = inset_axes(axes[0], width="35%", height="35%", loc="right")
    #
    # x1, x2 = 3, 4  # 放大尾部
    # for i, algo in enumerate(algos):
    #     mean = test_mean_accuracy_list[i]
    #     std = mean * 0.01
    #     axins.plot(round_axis, mean, color=colors[i])
    #     axins.fill_between(round_axis, mean - std, mean + std, color=colors[i], alpha=0.2)
    #
    # axins.set_xlim(x1 - 0.2, x2 + 0.2)
    # axins.set_ylim(0.25, 0.5)
    # axins.grid(True, linestyle="--", alpha=0.4)
    # axins.set_title("Zoom Tail", fontsize=9)
    # mark_inset(axes[0], axins, loc1=2, loc2=4, fc="none", ec="0.5")
    #
    # # ---------- Loss ----------
    # for i, algo in enumerate(algos):
    #     mean = test_mean_loss_list[i]
    #     std = mean * 0.0015
    #     axes[1].plot(round_axis, mean, label=algo, color=colors[i])
    #     axes[1].fill_between(round_axis, mean - std, mean + std,
    #                          color=colors[i], alpha=0.2)
    # axes[1].set_title(f"{title} Mean Test Loss")
    # axes[1].set_xlabel("Rounds")
    # axes[1].set_ylabel("Loss")
    # # axes[0].set_ylim([0, 1.4])
    # axes[1].grid(True, linestyle="--", alpha=0.5)
    # axes[1].legend(loc='upper right')
    #
    # # 局部放大图嵌在主图内部右下角
    # axins = inset_axes(axes[1], width="35%", height="35%", loc="right")
    #
    # x1, x2 = 3, 4  # 放大尾部
    # for i, algo in enumerate(algos):
    #     mean = test_mean_loss_list[i]
    #     std = mean * 0.01
    #     axins.plot(round_axis, mean, color=colors[i])
    #     axins.fill_between(round_axis, mean - std, mean + std, color=colors[i], alpha=0.2)
    #
    # axins.set_xlim(x1 - 0.2, x2 + 0.2)
    # axins.set_ylim(1.8, 2.0)
    # axins.grid(True, linestyle="--", alpha=0.4)
    # axins.set_title("Zoom Tail", fontsize=9)
    # mark_inset(axes[0], axins, loc1=2, loc2=4, fc="none", ec="0.5")
    #
    # plt.tight_layout()
    # plt.savefig("D:/PycharmProjects/FMGDA/FedCMOO/result/curve_paper_style_shaded.png", dpi=300, bbox_inches="tight")
    # plt.show()

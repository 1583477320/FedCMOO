import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import config
import os

# mode_list = [16,64,128,256]
# mode = "batch"

# mode_list = [5,10,30]
# mode = "client"

# mode_list = [1, 5, 10, 20, 50]
# mode = "local_round"

# mode_list = [1,0.1,0.01]
# mode = "lr"
mode = "algorithm"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ---------- 1. 读取数据 ----------
exp_configs = [
        config.base_config_set("base_config.json", experiment="MultiMNIST", algorithm="fsmgda_vr"),
        config.base_config_set('base_config.json', experiment="CIFAR10_MNIST", algorithm='fsmgda_vr'),
        config.base_config_set('base_config.json', experiment="CIFAR10_MNIST", algorithm='fsmgda_vr'),
        config.base_config_set('base_config.json', experiment="MNIST_FMNIST", algorithm= 'fsmgda_vr'),
        config.base_config_set('base_config.json', experiment = "MultiMNIST", algorithm= 'fsmgda_vr')
    ]

accuracy_df = pd.read_csv(f"D:/PycharmProjects/FMGDA/FedCMOO/result/{mode}/{exp_configs[0]['experiment']}_mean_accuracy.csv")
loss_df = pd.read_csv(f"D:/PycharmProjects/FMGDA/FedCMOO/result/{mode}/{exp_configs[0]['experiment']}_mean_loss.csv")

test_mean_accuracy_list = np.array([accuracy_df[col].tolist() for col in accuracy_df.columns])
test_mean_loss_list = np.array([loss_df[col].tolist() for col in loss_df.columns])

title = exp_configs[0]["experiment"]
# algos = [f"{mode}={mode_list[0]}", f"{mode}={mode_list[1]}", f'{mode}={mode_list[2]}']
mode_list=['FMSGDA-M-VR', 'FMSGDA', 'FedCMOO']
# mode_list=['FedCMOO', 'FMSGDA', 'FMSGDA-M-VR']
# mode_list=['FMSGDA',]
# algos = [f"{mode}={mode_list[i]}" for i in range(len(mode_list))]
# algos = ["fsmgda_vr"]

algos = [f"{mode_list[i]}" for i in range(len(mode_list))]
colors = ["#000000", "#E69F00", "#009E73",]  # 黑, 橙, 绿Q
# colors = ["#009E73", "#E69F00", "#000000",]
# colors = ["#000000", ]  # 黑, 橙, 绿Q
# colors = ["#0ef1cf", "#d7b9d5"]

# -------------------- Plot --------------------
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "lines.linewidth": 2
})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
round_axis = np.arange(1, len(test_mean_accuracy_list[0]) + 1)
# ---------- Accuracy ----------
for i, algo in enumerate(algos):
    mean = test_mean_accuracy_list[i]
    std = mean * 0.005
    axes[0].plot(round_axis, mean, label=algo, color=colors[i])
    axes[0].fill_between(round_axis, mean - std, mean + std,
                         color=colors[i], alpha=0.2)  # 阴影
axes[0].set_title(f"{title} Mean Test Accuracy")
axes[0].set_xlabel("Rounds")
axes[0].set_ylabel("Accuracy")
# axes[0].set_ylim([0.4, 0.8])
axes[0].set_ylim([0.7, 0.95])
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend(loc='lower right')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# 局部放大图嵌在主图内部右下角
# axins = inset_axes(axes[0], width="35%", height="35%", loc="right")
#
# x1, x2 = 350, 400  # 放大尾部
# for i, algo in enumerate(algos):
#     mean = test_mean_accuracy_list[i]
#     std = mean * 0.01
#     axins.plot(round_axis, mean, color=colors[i])
#     # axins.fill_between(round_axis, mean - std, mean + std, color=colors[i], alpha=0.2)
#
# axins.set_xlim(x1 - 0.2, x2 + 0.2)
# axins.set_ylim(0.83, 0.9)
# axins.grid(True, linestyle="--", alpha=0.4)
# # axins.set_title("Zoom Tail", fontsize=9)
# mark_inset(axes[0], axins, loc1=2, loc2=4, fc="none", ec="0.5")

# ---------- Loss ----------
for i, algo in enumerate(algos):
    mean = test_mean_loss_list[i]
    std = mean * 0.0015
    axes[1].plot(round_axis, mean, label=algo, color=colors[i])
    axes[1].fill_between(round_axis, mean - std, mean + std,
                         color=colors[i], alpha=0.2)
axes[1].set_title(f"{title} Mean Test Loss")
axes[1].set_xlabel("Rounds")
axes[1].set_ylabel("Loss")
# axes[0].set_ylim([0, 1.4])
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend(loc='upper right')

# 局部放大图嵌在主图内部右下角
# axins = inset_axes(axes[1], width="35%", height="35%", loc="right")
#
# for i, algo in enumerate(algos):
#     mean = test_mean_loss_list[i]
#     std = mean * 0.01
#     axins.plot(round_axis, mean, color=colors[i])
#     # axins.fill_between(round_axis, mean - std, mean + std, color=colors[i], alpha=0.2)
#
# axins.set_xlim(x1 - 0.2, x2 + 0.2)
# axins.set_ylim(0.235, 0.47)
# axins.grid(True, linestyle="--", alpha=0.4)
# # axins.set_title("Zoom Tail", fontsize=9)
# mark_inset(axes[1], axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.tight_layout()
plt.savefig(f"D:/PycharmProjects/FMGDA/FedCMOO/result/{mode}/{exp_configs[0]['experiment']}_curve_paper_style_shaded.png", dpi=300, bbox_inches="tight")
plt.show()
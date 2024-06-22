import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches
import yaml
from pathlib import Path
from settings_plot import *


def load_epochs(df_):

    epoch_dict_ = {}
    for row_ in df_.itertuples():

        path_ = Path(row_.path).parent.parent
        path_ = path_.joinpath("trainHistoryDict.pkl")
        model = row_.modelBS

        with open(path_, "rb") as f:
            train_history = pickle.load(f)
            epochs_ = len(train_history["loss"])
            epoch_dict_[model] = epochs_

    return epoch_dict_


path = "../../df_best_runs.pkl"
version = 3

with open(path, "rb") as f:
    best_models = pickle.load(f)
print(best_models.to_string())

epoch_dict = load_epochs(best_models)
df_epochs = pd.DataFrame(epoch_dict.items(), columns=["model", "epochs"])

result = {}
for row in best_models.itertuples():
    p = row.path
    p = Path(p).parent.parent
    p = p.joinpath("parameter.yml")

    with open(p, 'r') as stream:
        settings = yaml.safe_load(stream)
        time = settings["run_time"]
        # time = time.split(" ")[[0, 2]]
        time = list(map(time.split(" ").__getitem__, (0, 2)))
        time = int(time[0]) + int(time[1]) / 60

    result[row.modelBS] = time

df = pd.DataFrame(result.items(), columns=["model", "time"])
df = df.sort_values('time', ascending=True).reset_index(drop=True)

# separate bars by features
df["features"] = df.model.str.split(" ").str[1]
df["x"] = df.model.str.split(" ").str[::2].str.join(" ")
df["x"] = df.x.str.replace(" ", "\n")
print(df_epochs)
df = df.merge(df_epochs, on="model")

###########
# RUNTIME #
###########

df["features"] = np.where(df.features.str.contains("ESF"), "With Static Features", "Without Static Features")

df_epochs = df.copy()

# sort x axis
x_axis_order = ["CNN\nBS=256", "CNN\nBS=2048", "LSTM\nBS=256", "LSTM\nBS=2048", "GRU\nBS=256", "GRU\nBS=2048"]

fig = plt.figure(figsize=(20, 8))
ax = sns.barplot(data=df, x='x', y='time', hue='features',
                 order=x_axis_order, ec="#363636",
                 hue_order=["With Static Features", "Without Static Features"])
plt.setp(ax.patches, linewidth=1.5)
ax.set(ylim=(0, 30))

for i in ax.containers:
    ax.bar_label(i, fmt='%.1f', label_type="center", size=14, weight="bold", color="white")

# legend
handles, labels = ax.get_legend_handles_labels()
handles.append(mpatches.Patch(color='none', linestyle="none", label="test"))
ax.legend(handles=handles, labels=labels)
sns.move_legend(ax, "upper center", bbox_to_anchor=(.5, 0.98), ncol=2, shadow=False, title=None, frameon=~False)

ax.set_ylabel("Runtime [min]", labelpad=20, weight="bold")
ax.set_xlabel("Model architecture - batch size", labelpad=20, weight="bold")
fig.tight_layout()

sns.despine(right=True, left=True)
####################
# df.drop("epochs", axis=1, inplace=True)
df["batch size"] = df.x.str.split("=").str[-1]
df.drop("x", axis=1, inplace=True)
df["model"] = df.model.str.split(" ").str[0]
# df["model"] = df.model + df["batch size"].astype(str)
# df.drop("batch size", axis=1, inplace=True)
# df.drop("time", axis=1, inplace=True)
df = df.reset_index(drop=True)
df.sort_values('epochs', ascending=True).reset_index(drop=True)
df = df.round(1)

plt.show()
fig.savefig(os.path.join("../../results/Compare_model_runtime_v%s.pdf" % version), format="pdf",
            bbox_inches="tight")
plt.close(fig)

##########
# EPOCHS #
##########

# sort x axis ascending by time with static features
df_epochs = df_epochs.sort_values('epochs', ascending=True).reset_index(drop=True)
grouped = df_epochs[df_epochs.features == "With Static Features"].reset_index(drop=True)
sorted_group = grouped['x'].tolist()
print(df_epochs.to_string())

fig = plt.figure(figsize=(20, 8))
ax = sns.barplot(data=df_epochs, x='x', y='epochs', hue='features',
                 order=sorted_group, ec="#363636",
                 hue_order=["With Static Features", "Without Static Features"])
plt.setp(ax.patches, linewidth=1.5)
ax.set(ylim=(0, 55))

for i in ax.containers:
    ax.bar_label(i, fmt='%.0f', label_type="center", size=14, weight="bold", color="white")

# legend
handles, labels = ax.get_legend_handles_labels()
handles.append(mpatches.Patch(color='none', linestyle="none", label="test"))
ax.legend(handles=handles, labels=labels)
sns.move_legend(ax, "upper center", bbox_to_anchor=(.5, 1.02), ncol=2, shadow=False, title=None, frameon=~False)

ax.set_ylabel("Epochs", labelpad=20, weight="bold")
ax.set_xlabel("Model architecture - batch size", labelpad=20, weight="bold")
fig.tight_layout()

sns.despine(right=True, left=True)

# plt.show()

fig.savefig(os.path.join("../../results/Compare_epochs_v%s.pdf" % version), format="pdf",
            bbox_inches="tight")
plt.close(fig)

################
# EPOCHS Paper #
################

# Filter for bs = 256 and with static features
df_epochs["bs"] = df_epochs.model.str.split("=").str[-1].astype(int)
df_epochs = df_epochs.query("features == 'With Static Features' & bs == 256")
print(df_epochs.to_string())

# sort x axis ascending by time with static features
df_epochs = df_epochs.sort_values('epochs', ascending=True).reset_index(drop=True)
sorted_group = df_epochs['x'].tolist()

fig = plt.figure(figsize=(10, 8))
ax = sns.barplot(data=df_epochs, x='x', y='epochs',
                 order=sorted_group, ec="#363636", palette=["#176D9C", "#2EABB8", "#57D2C3"])
plt.setp(ax.patches, linewidth=1.5)
ax.set(ylim=(0, 35))

for i in ax.containers:
    ax.bar_label(i, fmt='%.0f', label_type="center", size=14, weight="bold", color="white")

# legend
# handles, labels = ax.get_legend_handles_labels()
# handles.append(mpatches.Patch(color='none', linestyle="none", label="test"))
# ax.legend(handles=handles, labels=labels)
# sns.move_legend(ax, "upper center", bbox_to_anchor=(.5, 1.02), ncol=2, shadow=False, title=None, frameon=~False)

ax.set_ylabel("Epochs", labelpad=20, weight="bold")
ax.set_xlabel("Model architecture", labelpad=20, weight="bold")
fig.tight_layout()

sns.despine(right=True, left=True)

plt.show()

fig.savefig(os.path.join("../../results/Compare_epochs_v%s.pdf" % version), format="pdf",
            bbox_inches="tight")
plt.close(fig)
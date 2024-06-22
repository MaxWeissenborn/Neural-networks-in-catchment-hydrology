import os
import pickle
import pandas as pd
from timeit import default_timer as timer
import sys
from pathlib import Path
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from matplotlib.markers import MarkerStyle
import matplotlib.patches as mpatches
from functions_and_settings import pathlist, scantree
from scripts.custom_functions.general import path_join
import decimal
from settings_plot import *

version = "1"
data = {}
for path in pathlist:
    if not os.path.exists(path):
        continue
    for entry in scantree(sys.argv[1] if len(sys.argv) > 1 else path):
        if entry.name.endswith("metrics_test-results.pkl") and entry.is_file():
            # print(entry.name, entry.path)
            if "lr_improve" not in entry.path:
                with open(entry.path, "rb") as f:
                    metrics = pickle.load(f)
                    kge_list = [i["kge"] for i in metrics]
                    # data[entry.path] = kge_list
                    if len(kge_list) != 35:
                        print(len(kge_list))
                        print(entry.path)
                    else:
                        # if "(25)" in entry.path:
                        data[entry.path] = kge_list

# print(data)
df = pd.DataFrame(data)
raw_df = df.copy()
means = df.mean(axis=0)
df = pd.DataFrame(means, columns=["mean"]).reset_index().rename(columns={"index": 'path'})

df["BS"] = df.path.str.split("/").str[-3].str.split().str[-1].astype(int)
df["model"] = df.path.str.split("/").str[-6]
df["epochs"] = df.path.str.split("/").str[-4].str.split().str[-1].astype(int)

df["modelBS"] = df.model + " BS=" + df.BS.astype(str)
df = df.query("BS==2048 | BS == 256")
df = df.dropna().reset_index().copy()

# filter df by unique model/batch size combinations with this line
df = df.sort_values('mean', ascending=False).drop_duplicates(["modelBS"]).reset_index()
pickle.dump(df, open(path_join([Path.cwd().parent.parent, "df_best_runs.pkl"]), "wb"))

print(df.to_string())

df = df.copy()

##########################
# prepare df for seaborn #
##########################

# get data for violin plot
res_dict = {}
for row in df.itertuples():
    bs = row.BS
    res_file = os.path.join(Path(row.path).parent, "evaluation_metrics_test-results.pkl")
    if os.path.isfile(os.path.abspath(res_file)):
        with open(res_file, "rb") as f:
            metrics_list = pickle.load(f)
            kge_list = [metric_dict["kge"] for metric_dict in metrics_list]
            res_dict[row.modelBS] = kge_list #sorted(kge_list)

df_all = pd.DataFrame(res_dict)

df_all["count"] = df_all.index + 1
df_melt = pd.melt(df_all, id_vars="count", value_vars=df_all.columns)

df_melt["features"] = np.where(df_melt.variable.str.contains("ESF"), "With Static Features", "Without Static Features")
df_melt.variable = df_melt.variable.str.replace('\+ESF', '', regex=True)
df_melt.variable = df_melt.variable.str.replace('\-SF', '', regex=True)


#################
# Plotting Part #
#################

# sort x axis
x_axis_order = ["CNN\nBS=256", "CNN\nBS=2048", "LSTM\nBS=256", "LSTM\nBS=2048", "GRU\nBS=256", "GRU\nBS=2048"]

# set line break for x ticks
df_melt["variable"] = df_melt.variable.str.replace("  ", "\n", regex=False)

fig = plt.figure(figsize=(20, 8))
ax = sns.violinplot(data=df_melt, x="variable", y="value", hue="features", density_norm="count",
                    hue_order=["With Static Features", "Without Static Features"],
                    cut=0, inner="quartile", split=True,
                    order=x_axis_order)

means1 = df_melt[df_melt.features == "With Static Features"].groupby('variable', sort=False)['value'].mean()
means2 = df_melt[df_melt.features == "Without Static Features"].groupby('variable', sort=False)['value'].mean()
means1 = pd.DataFrame(means1)
means2 = pd.DataFrame(means2)
means1["group"] = "With Static Features"
means2["group"] = "Without Static Features"

df = pd.concat([means1, means2])
df["name"] = df.index.str.replace(r"\n", " - ", regex=False)

plt.draw()  # create plot to access elements with the next line
xtick_loc = {v.get_text(): v.get_position()[0] for v in ax.get_xticklabels()}  # get coordinates of quartile lines
ytick_loc = {'CNN\nBS=256': [1.02, 0.99], 'LSTM\nBS=256': [0.985, 0.97],
             'CNN\nBS=2048': [0.98, 1.01], 'GRU\nBS=256': [0.99, 0.99],
             'GRU\nBS=2048': [0.99, 1.01], 'LSTM\nBS=2048': [1.01, 0.99]}

for idx, row in enumerate(df.itertuples()):
    x = xtick_loc[row.Index]
    if row.group == "With Static Features":
        y = row.value * ytick_loc[row.Index][0]
        if decimal.Decimal(str(round(row.value, 2))).as_tuple().exponent == -2:
            t = ax.text(x - 0.20, y, round(row.value, 2), size=15, weight='bold', color="white")
        else:
            t = ax.text(x - 0.15, y, round(row.value, 2), size=15, weight='bold', color="white")
    else:
        y = row.value * ytick_loc[row.Index][1]
        t = ax.text(x + 0.04, y, round(row.value, 2), size=15, weight='bold', color="white")

    t.set_bbox(dict(facecolor='grey', alpha=0.7, edgecolor='grey'))


# plotting the means with a dot
sns.stripplot(data=df, x="variable", y="value", hue="group", s=12, legend=None, jitter=False,
              linewidth=.7, edgecolor="white", ax=ax)

# this part was for the case when 2 means of one violin are equal
# sns.stripplot(data=means.query("name == 'CNN - BS=256'"), x="variable", y="value", s=12, legend=None, jitter=False,
#               ax=ax, marker=MarkerStyle("o", fillstyle="left"))
# sns.stripplot(data=means.query("name == 'CNN - BS=256'"), x="variable", y="value", s=12, legend=None, jitter=False,
#               linewidth=.7, marker="o", fc="none",
#               edgecolor="white", ax=ax, )

ax.set(ylim=(-0.3, 1.1))
# Show the minor grid as well. Style it in very light gray as a thin, dotted line.
ax.yaxis.set_minor_locator(AutoMinorLocator(2))  # number of subdivisions between major ticks
ax.grid(which='minor', color='#7E7B7B', linestyle=':', linewidth=.35)

# legend
handles, labels = ax.get_legend_handles_labels()
handles.append(mpatches.Patch(color='none', linestyle="none", label="test"))
ax.legend(handles=handles, labels=labels)
sns.move_legend(ax, "upper center", bbox_to_anchor=(.5, 1.013), ncol=2, shadow=False, title=None, frameon=True)

ax.set_ylabel("KGE", labelpad=20, weight="bold")
ax.set_xlabel("Model architecture - batch size", labelpad=20, weight="bold")
fig.tight_layout()

sns.despine(right=True, left=True)

# fig.savefig(os.path.join("../../results/Modelcomp-Static.vs.NoStatic-Violinplot_v%s.pdf" % version), format="pdf",
#             bbox_inches="tight")
# plt.close(fig)

plt.show()

# kann dann weg #todo
df = pickle.load(open(path_join([Path.cwd().parent.parent, "df_best_runs.pkl"]), "rb"))
max = raw_df.max(axis=0)
min = raw_df.min(axis=0)
df_max = pd.DataFrame(max, columns=["max"]).reset_index().rename(columns={"index": 'path'})
df_min = pd.DataFrame(min, columns=["min"]).reset_index().rename(columns={"index": 'path'})
df = pd.merge(df, df_max, how="inner", on="path")
df = pd.merge(df, df_min, how="inner", on="path")
df["range"] = df["max"] - df["min"]
# df = df.query("model.str.endswith('-SF')")
print(df.sort_values('range', ascending=False))


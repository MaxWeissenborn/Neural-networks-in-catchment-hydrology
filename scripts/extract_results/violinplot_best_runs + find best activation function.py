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
from scripts.extract_results.functions_and_settings import pathlist, scantree
from scripts.custom_functions.general_for_db_speed_test import path_join
import decimal
from scripts.extract_results.settings_plot import *

version = "25"
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
# pickle.dump(df, open(path_join([Path.cwd().parent.parent, "df_best_runs_all_kge.pkl"]), "wb"))
pickle.dump(df, open(path_join([Path.cwd(), "df_best_runs_all_kge.pkl"]), "wb"))

means = df.mean(axis=0)
means = pd.DataFrame(means, columns=["mean"]).reset_index().rename(columns={"index": 'path'})
# means["modelBS"] = means.path.str.split("/").str[-6] + " - " + means.path.str.split("/").str[-3]
means["BS"] = means.path.str.split("/").str[-3].str.split().str[-1].astype(int)
means["model"] = means.path.str.split("/").str[-6]
means["epochs"] = means.path.str.split("/").str[-4].str.split().str[-1].astype(int)
means = means.replace({"model": {"CNN_Deep +ESF": "CNN +ESF", "CNN_Deep -SF": "CNN -SF"}})

means["modelBS"] = means.model + " BS=" + means.BS.astype(str)
means = means.query("BS==2048 | BS == 256")
means = means.dropna().reset_index().copy()
# print(means.query("epochs==88").to_string())
print(means.to_string())
means = means.query("epochs>=62")
df_melt = pd.melt(means, id_vars="epochs", value_vars="mean")
print(df_melt)

epochs = []
value = []
for index, row in df_melt.iterrows():
    epochs.append(row['epochs'])
    value.append(row['value'])

print("data= {'epochs': %s,\n'value': %s}" % (epochs, value))

# data= {'epochs': [62, 63, 64, 65, 66, 67, 68, 62, 63, 64, 65, 66, 67, 68, 62, 63, 64, 65, 66, 67, 68, 62, 63, 64, 65, 66, 67, 68, 62, 63, 64, 65, 66, 67, 68, 62, 63, 64, 65, 66, 67, 68, 62, 63, 64, 65, 66, 67, 68, 62, 63, 64, 65, 66, 67, 68, 62, 63, 64, 65, 66, 67, 68, 62, 63, 64, 65, 67, 68],
# 'value': [0.5403226609563172, 0.23879261489859052, 0.2065494039663671, 0.2098798066544705, 0.2614686219793561, 0.43680225043356635, 0.2222405974854495, 0.40147504779572674, 0.4814138094151069, 0.4578304280624048, 0.1960999978419161, 0.39976486962191327, 0.5014639444103237, 0.4800109962327052, 0.6008748018221786, 0.1305588501191859, 0.03123097919145981, 0.13774706478823617, 0.034384405408350194, 0.5756144764579639, 0.08696491966588739, 0.6405520607479757, 0.03671175516039281, 0.12460874898410572, 0.09177676337444804, 0.1587626978730697, 0.6030500133405773, 0.3815343011797176, 0.5895639223940325, 0.36888822435566704, 0.14444165010151894, 0.09097628943250986, 0.15428174281064358, 0.6654515493651908, 0.4228902438931862, 0.5632890516105945, 0.2179355395949716, 0.15161554364644525, 0.2071709431929074, 0.27423646973327503, 0.4429101041246471, 0.24811450450976932, 0.6364738441783615, 0.24397896103869132, 0.21573843988828373, 0.24266006544598737, 0.23381516466910535, 0.6242732613828517, 0.18799671659505066, 0.6072740745831939, 0.21270302866282095, 0.24778868492556316, 0.2806697282613287, 0.1958910028309859, 0.5159105157589889, 0.2015775762723086, 0.45346002108070643, 0.23134920593756853, 0.02657677370890841, -0.02632257131215408, 0.2481434248899687, 0.6074012397386166, -0.0041165008476349045, 0.5317136399077808, 0.16057419697955147, 0.21393545783109755, 0.1831951508664548, 0.6807887302991984, 0.11333422690815939]}
data= {'epochs': [80, 81, 82, 80, 81, 82, 80, 81, 82, 80, 81, 82, 80, 81, 82, 80, 81, 82, 80, 81, 82, 80, 81, 82, 80, 81, 82, 80, 81, 82, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83],
'value': [0.7566914598503293, 0.7428996702353646, 0.7463078777419193, 0.7433955267234227, 0.7649851091038162, 0.7170683598099409, 0.6629206298652457, 0.7468386528130875, 0.7633068112743195, 0.7858656714258992, 0.7816696038516059, 0.7778324956990309, 0.7292912051822694, 0.7218538118377771, 0.7638069130632616, 0.7355818851388619, 0.7549241733497688, 0.749126606126285, 0.7515205144955565, 0.7344787926170833, 0.7477098403060347, 0.7517913917764214, 0.7580049307450273, 0.7740960167666245, 0.7763040507507946, 0.7709264708249159, 0.7638086506650006, 0.7718937100284088, 0.7875397165960067, 0.7883328678272902, 0.722777386898129, 0.6967089463314029, 0.7337065046292882, 0.7293958223003157, 0.7156821289656303, 0.7065275786473745, 0.7280884715885754, 0.7164816532356367, 0.7467737367551992, 0.6874142698046709]}


# Create DataFrame
df = pd.DataFrame(data)
df = df.query("epochs!=66")
df = df.query("epochs!=68")
df.epochs = np.where(df.epochs == 62, "Whole data set", df.epochs)
df.epochs = np.where(df.epochs == "63", "Generator case 1", df.epochs)
df.epochs = np.where(df.epochs == "64", "Generator case 2", df.epochs)
df.epochs = np.where(df.epochs == "65", "Generator case 3", df.epochs)
df.epochs = np.where(df.epochs == "67", "Generator case 4", df.epochs)
print(df[df.epochs == "82"])
fig = plt.figure(figsize=(20, 8))
ax = sns.boxplot(data=df, y="epochs", x="value", orient="h")
ax.set_xlabel("R²", labelpad=20, weight="bold")
ax.set_ylabel("", labelpad=20, weight="bold")
# ax.set(ylim=(0.4, 1.1))
plt.show()
fig.savefig(os.path.join("test.png"), format="png",
            bbox_inches="tight")
plt.close(fig)


0 / 0
# filter df by unique model/batch size combinations with this line
# means = means.sort_values('mean', ascending=False).drop_duplicates(["modelBS"]) # todo change this back
print(means.groupby("epochs")["mean"].max())
means = means.sort_values('mean', ascending=False).drop_duplicates(["epochs"])
# pickle.dump(means, open(path_join([Path.cwd().parent.parent, "df_best_runs.pkl"]), "wb"))
pickle.dump(means, open(path_join([Path.cwd(), "df_best_runs.pkl"]), "wb"))

print(means.to_string())
0 / 0
df = means.copy()

##########################
# prepare df for seaborn #
##########################

res_dict = {}
for row in df.itertuples():
    bs = row.BS
    res_file = os.path.join(Path(row.path).parent, "KGE_test-results.pkl")
    if os.path.isfile(os.path.abspath(res_file)):
        with open(res_file, "rb") as f:
            kge_list = pickle.load(f)
            res_dict[row.modelBS] = sorted(kge_list)

df = pd.DataFrame(res_dict)

# remove new tested model
drop_cols = []
for col in df.columns:
    if col.startswith("CNN_LSTM"):
        drop_cols.append(col)

df = df.drop(drop_cols, axis=1)

df["count"] = df.index + 1
df_melt = pd.melt(df, id_vars="count", value_vars=df.columns)

df_melt["features"] = np.where(df_melt.variable.str.contains("ESF"), "With Static Features", "Without Static Features")
df_melt.variable = df_melt.variable.str.replace('\+ESF', '', regex=True)
df_melt.variable = df_melt.variable.str.replace('\-SF', '', regex=True)

# sort x axis
x_axis_order = ["CNN\nBS=256", "CNN\nBS=2048", "LSTM\nBS=256", "LSTM\nBS=2048", "GRU\nBS=256", "GRU\nBS=2048"]

#################
# Plotting Part #
#################

# set line break for x ticks

df_melt["variable"] = df_melt.variable.str.replace("  ", "\n", regex=False)
fig = plt.figure(figsize=(20, 8))
ax = sns.violinplot(data=df_melt, x="variable", y="value", hue="features",
                    hue_order=["With Static Features", "Without Static Features"],
                    cut=0, inner="quartile", split=True,
                    order=x_axis_order)

means1 = df_melt[df_melt.features == "With Static Features"].groupby('variable', sort=False)['value'].mean()
means2 = df_melt[df_melt.features == "Without Static Features"].groupby('variable', sort=False)['value'].mean()
means1 = pd.DataFrame(means1)
means2 = pd.DataFrame(means2)
means1["group"] = "With Static Features"
means2["group"] = "Without Static Features"

means = pd.concat([means1, means2])
means["name"] = means.index.str.replace(r"\n", " - ", regex=False)

plt.draw()  # create plot to access elements with the next line
xtick_loc = {v.get_text(): v.get_position()[0] for v in ax.get_xticklabels()}  # get coordinates of quartile lines
ytick_loc = {'CNN\nBS=256': [1.02, 0.99], 'LSTM\nBS=256': [0.985, 0.97],
             'CNN\nBS=2048': [0.98, 1.01], 'GRU\nBS=256': [0.99, 0.99],
             'GRU\nBS=2048': [0.99, 1.01], 'LSTM\nBS=2048': [1.01, 0.99]}

for idx, row in enumerate(means.itertuples()):
    x = xtick_loc[row.Index]
    if row.group == "With Static Features":
        y = row.value * ytick_loc[row.Index][0]
        if decimal.Decimal(str(round(row.value, 2))).as_tuple().exponent == -2:
            ax.text(x - 0.20, y, round(row.value, 2), size=15, weight='bold', color="white")
        else:
            ax.text(x - 0.15, y, round(row.value, 2), size=15, weight='bold', color="white")
    else:
        y = row.value * ytick_loc[row.Index][1]
        ax.text(x + 0.04, y, round(row.value, 2), size=15, weight='bold', color="white", )
        # path_effects=[pe.withStroke(linewidth=2, foreground="black")])

# plotting the means with a dot
sns.stripplot(data=means, x="variable", y="value", hue="group", s=12, legend=None, jitter=False,
              linewidth=.7, edgecolor="white", ax=ax)

# this part was for the case when 2 means of one violin are equal
# sns.stripplot(data=means.query("name == 'CNN - BS=256'"), x="variable", y="value", s=12, legend=None, jitter=False,
#               ax=ax, marker=MarkerStyle("o", fillstyle="left"))
# sns.stripplot(data=means.query("name == 'CNN - BS=256'"), x="variable", y="value", s=12, legend=None, jitter=False,
#               linewidth=.7, marker="o", fc="none",
#               edgecolor="white", ax=ax, )

ax.set(ylim=(0, 1.1))
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

fig.savefig(os.path.join("../../results/Modelcomp-Static.vs.NoStatic-Violinplot%s.pdf" % version), format="pdf",
            bbox_inches="tight")
plt.close(fig)

# plt.show()

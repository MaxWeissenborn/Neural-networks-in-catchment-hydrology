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
from scripts.custom_functions.general_for_db_speed_test import get_river_ids
import decimal
from settings_plot import *
import spotpy
from datetime import timedelta
from matplotlib.ticker import MultipleLocator

model = "LSTM"  # check uncheck the wanted model
# model = "CNN"
prefix = "1"
filename = "lineplot_data.pkl"


def generate_data():
    data = {}
    for path in pathlist:
        for entry in scantree(sys.argv[1] if len(sys.argv) > 1 else path):
            if entry.name.endswith("KGE_test-results.pkl") and entry.is_file():
                # print(entry.name, entry.path)
                if "lr_improve" not in entry.path:
                    with open(entry.path, "rb") as f:
                        kge_list = pickle.load(f)
                        # data[entry.path] = kge_list
                        if len(kge_list) != 35:
                            print(len(kge_list))
                            print(entry.path)
                        else:
                            # if "(25)" in entry.path:
                            data[entry.path] = kge_list

    # print(data)
    df = pd.DataFrame(data)

    model_features_list = ['CNN +ESF', 'LSTM +ESF']
    output = {}
    for model_features in model_features_list:
        model = model_features.split(" ")[0]

        means = df.mean(axis=0)
        means = pd.DataFrame(means, columns=["mean"]).reset_index().rename(columns={"index": 'path'})
        means["BS"] = means.path.str.split("/").str[-3].str.split().str[-1].astype(int)
        means["model"] = means.path.str.split("/").str[-6]
        means["epochs"] = means.path.str.split("/").str[-4].str.split().str[-1].astype(int)
        means = means.replace({"model": {"CNN_Deep +ESF": "CNN +ESF", "CNN_Deep -SF": "CNN -SF"}})

        selection = means[means.model == model_features]
        selection = selection.query("BS==256")
        best_run = selection.loc[selection["mean"].idxmax()]
        # print(means.to_string())
        # print(best_run.path)
        path = Path(best_run.path).parent

        with open(path.joinpath("test_observation_unscaled.pkl"), "rb") as f:
            obs = pickle.load(f)

        with open(path.joinpath("test_prediction_unscaled.pkl"), "rb") as f:
            pred = pickle.load(f)

        river_ids, date_index = get_river_ids(
            "../../preprocessing/output/+ESF - NO NA - VALIDATION NOT FOR TRAIN - Complete River Data as Dataframe - 1997 - 2002.pkl",
            with_index=True)

        n = 35
        split = int(len(obs) / n)

        results = {}
        data = {}

        for i in range(n):
            y_sample = obs[i * split:(i + 1) * split]
            p_sample = pred[i * split:(i + 1) * split]

            kge = spotpy.objectivefunctions.kge(y_sample, p_sample)
            results[river_ids[i]] = kge
            data[river_ids[i]] = {"obs": y_sample, "pred": p_sample}

        best_catchment = max(results, key=lambda k: results[k])
        worst_catchment = min(results, key=lambda k: results[k])
        print(model)
        print(results.keys())
        print(worst_catchment)

        new_best_data = zip(date_index, data[best_catchment]["obs"], data[best_catchment]["pred"])

        df_neu = pd.DataFrame(new_best_data, columns=["date", "obs", "pred"])
        df_neu.date = pd.to_datetime(df_neu.date)
        df_neu = df_neu.set_index("date", drop=True)
        df_neu = df_neu.astype('float32')

        output[model] = {"best": df_neu}

        new_worst_data = zip(date_index, data[worst_catchment]["obs"], data[worst_catchment]["pred"])

        df_neu = pd.DataFrame(new_worst_data, columns=["date", "obs", "pred"])
        df_neu.date = pd.to_datetime(df_neu.date)
        df_neu = df_neu.set_index("date", drop=True)
        df_neu = df_neu.astype('float32')

        output[model]["worst"] = df_neu

    pickle.dump(output, open(filename, "wb"))


if not os.path.isfile(filename):
    generate_data()
with open("lineplot_data.pkl", "rb") as f:
    data = pickle.load(f)



if model == "CNN":
    df = data["CNN"]["best"]
    df.index = df.index.shift(26, freq='D')
    df_melt = df.reset_index().melt("date", var_name='cols', value_name='vals')
elif model == "LSTM":
    df = data["LSTM"]["best"]
    df = df.iloc[26:, :].copy()
    df_melt = df.reset_index().melt("date", var_name='cols', value_name='vals')
else:
    0 / 0


fig = plt.figure(figsize=(20, 9), layout="tight")
spec = fig.add_gridspec(ncols=2, nrows=2)

ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[1, 0], sharex=ax1)
ax3 = fig.add_subplot(spec[:, 1])

l1 = sns.lineplot(data=df_melt, x="date", y="vals", hue="cols", palette=["#176D9C", "#DE8F05"], linewidth=1, ax=ax1)
# make all negative values red
for idx, row in df.iterrows():
    if row.pred < 0:
        sns.lineplot(x=[idx, idx + timedelta(days=1)],
                     y=[row.pred, df.loc[idx + timedelta(days=0), 'pred']], color="red", linewidth=1, ax=ax1)

l1.set_ylim(-0.5, top=27)
l1.tick_params(labelbottom=False)
l1.set_xlabel(xlabel=None)
l1.set_ylabel("Discharge [mm]", labelpad=20, weight="bold")
l1.text(df.index[0], 25.5, "a)", size=16, weight='bold', color="black", )

df["res"] = df.pred - df.obs
average = np.mean(df.res)
l2 = sns.lineplot(data=df, x="date", y="res", linewidth=1, color="#108010", ax=ax2)
l2.set_ylabel("Residuals [mm]", labelpad=20, weight="bold")
l2.set_xlabel("Date", labelpad=20, weight="bold")
l2.set_ylim(-9, top=9)
l2.set_xlim(df.index[0], df.index[-1])
l2.yaxis.set_major_locator(MultipleLocator(2))
l2.text(df.index[0], 8.5, "b)", size=16, weight='bold', color="black", )
l2.text(df.index[0] + timedelta(days=365 * 2.15), 6.6, "Mean residual: %s" % average.round(2), size=20, color="black",
        bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round, pad=0.35, rounding_size=0.2', linewidth=0.5))

# legend
kge = spotpy.objectivefunctions.kge(df.obs, df.pred)
handles, labels = ax1.get_legend_handles_labels()
if labels[0] == "obs":
    labels = ["Observation", "Prediction      KGE: %s" % kge.round(2)]

# handles.append(mpatches.Patch(color='none', linestyle="none", label="test"))
ax1.legend(handles=handles, labels=labels)
sns.move_legend(ax1, "upper center", bbox_to_anchor=(.5, 1.04), ncol=4, shadow=False, title=None, frameon=~False,
                framealpha=1)

# regression plot
rsquare = spotpy.objectivefunctions.rsquared(df.obs, df.pred)
s = sns.regplot(data=df, x="obs", y="pred", ax=ax3, line_kws={"color": "red"}, color="#176D9C")
s.set(aspect='equal')
s.yaxis.set_major_locator(MultipleLocator(2))
s.xaxis.set_major_locator(MultipleLocator(2))
s.set_ylabel("Prediction [mm]", labelpad=20, weight="bold")
s.set_xlabel("Observation [mm]", labelpad=20, weight="bold")
s.set_ylim(-0.5, 24)
s.set_xlim(-0.5, 24)
s.text(0, 24.2, "c)", size=16, weight='bold', color="black", )

# legend regression plot
handles = [mpatches.Patch(color='red', alpha=0.2, label="R²: %s" % np.round(rsquare, 2))]
s.legend(handles=handles, labels=["Confidence interval: 95%%     R²: %s" % np.round(rsquare, 2)])
sns.move_legend(s, "upper center", bbox_to_anchor=(.5, 1.006), ncol=4, shadow=False, title=None, frameon=True,
                framealpha=1)

# # layout
fig.tight_layout()
sns.despine(bottom=True, left=True, )

plt.show()
fig.savefig(os.path.join("../../results/Lineplot_%s_best_run_v%s.pdf" % (model, prefix)), format="pdf",
            bbox_inches="tight")
plt.close(fig)

import pickle
import os
from pathlib import Path
import functions_and_settings
import pandas as pd
from scripts.custom_functions.general import calc_evaluation_metrics
import matplotlib.patches as mpatches
from settings_plot import *

version = 3
root = Path.cwd().parent.parent
os.chdir(root)
with open("df_best_runs.pkl", "rb") as f:
    df = pickle.load(f)

# filter out models that do not have ESF and BS = 256
df_ = df.query("BS == 256 and model.str.endswith('+ESF')")
print(df_)

bins = None
results = {}
for row in df_.itertuples():
    model = row.model
    path = Path(*Path(row.path).parent.parts[2:])
    pred_path = os.path.join(path, "test_prediction_unscaled.pkl")
    obs_path = os.path.join(path, "test_observation_unscaled.pkl")
    res = {}
    for file in [pred_path, obs_path]:
        if os.path.isfile(os.path.abspath(file)):
            with open(file, "rb") as f:

                data = pickle.load(f)
                if "prediction" in file:
                    res["pred"] = data
                else:
                    res["obs"] = data
                    max_index = len(data)

    df = pd.DataFrame(res)
    split = max_index / 35

    result = {}
    for i in range(35):

        df_ = df.iloc[int(i * split):int((i + 1) * split)]
        groups = df_.groupby(
            pd.qcut(df_.obs, 4, labels=["Lowest flows Q1", "Lower flows Q2", "Higher flows Q3", "Highest flows Q4"]))
        for key, grp in groups:
            if key not in result:
                result[key] = []
            metrics = calc_evaluation_metrics(grp.obs, grp.pred)
            result[key].append(metrics)

    results[model] = {}
    for key in result:
        results[model][key] = pd.DataFrame(result[key])

# Reshape the Data for both metrics into a single DataFrame
data = []
for model_name, flows in results.items():
    for flow_type, df in flows.items():
        for kge_value, pbias_value in zip(df['kge'], df['pbias']):
            data.append([model_name, flow_type, kge_value, 'KGE'])
            data.append([model_name, flow_type, pbias_value, 'PBIAS'])

# Convert list to DataFrame
reshaped_data = pd.DataFrame(data, columns=['Model', 'Flow', 'Value', 'Metric'])
reshaped_data["x"] = reshaped_data.Flow.str.split(' ').str[-1]

pattern = r'(Q\d+)'
replacement = r'(\1)'
reshaped_data.Flow = reshaped_data.Flow.str.replace(pattern, replacement, regex=True)

reshaped_data['Model'] = reshaped_data['Model'].str.replace('\+ESF', 'with Static Features', regex=True)

# Create the 2x3 grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13.5, 9), sharey='row', sharex='col')

sns.set_palette("colorblind")
# plotting
metrics = ['KGE', 'PBIAS']
for i, metric in enumerate(metrics):
    for j, model in enumerate(reshaped_data['Model'].unique()):
        ax = axes[i, j]
        sns.violinplot(x='x', y='Value', cut=0, inner="quartile", density_norm="count", hue="Flow",
                       data=reshaped_data[(reshaped_data['Model'] == model) & (reshaped_data['Metric'] == metric)],
                       ax=ax)
        # plotting the means with a dot
        data = reshaped_data[(reshaped_data['Model'] == model) & (reshaped_data['Metric'] == metric)].groupby(
            'x').mean().reset_index()

        sns.stripplot(data=data,
                      x="x", y='Value', s=8, legend=None, jitter=False,
                      linewidth=.7, color="black", ax=ax)

        for idx, row in enumerate(data.itertuples()):
            x = idx
            y = row.Value
            t = ax.text(x + 0.15, y, round(row.Value, 2), size=10, weight='bold', color="white")
            t.set_bbox(dict(facecolor='grey', alpha=0.7, edgecolor='grey'))

        if i == 0:
            ax.set_title(model)
            ax.set_ylim(-6, 1)
        else:
            ax.set_ylim(-100, 500)
        ax.set_xlabel('')
        if j == 0:
            ax.set_ylabel(metric, labelpad=20, weight="bold")
        else:
            ax.set_ylabel('')
        ax.get_legend().remove()

axes[1, 1].set_xlabel("Flow segments", labelpad=20, weight="bold")
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(mpatches.Patch(color='none', linestyle="none", label="test"))
fig.legend(handles=handles, labels=labels, loc='center')
sns.move_legend(fig, loc="center", bbox_to_anchor=(.535, 0.5), ncol=2, shadow=False, title=None, frameon=True)

plt.tight_layout()

fig.savefig(os.path.join("results/Modelcomp-percentiles-Violinplot_v%s.pdf" % version), format="pdf",
            bbox_inches="tight")
plt.close(fig)

plt.show()

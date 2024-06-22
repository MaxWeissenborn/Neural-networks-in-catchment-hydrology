import pickle
from pathlib import Path
from scripts.custom_functions.general import path_join
from functions_and_settings import *

df = pickle.load(open(path_join([Path.cwd().parent.parent, "df_best_runs.pkl"]), "rb"))

# get data for violin plot
res_dict = {}
for row in df.itertuples():
    bs = row.BS
    res_file = os.path.join(Path(row.path).parent, "evaluation_metrics_test-results.pkl")
    if os.path.isfile(os.path.abspath(res_file)):
        res_dict[row.modelBS] = {}
        with open(res_file, "rb") as f:
            metrics_list = pickle.load(f)
            for metric in metrics_list[0].keys():
                res_dict[row.modelBS][metric] = np.mean([metric_dict[metric] for metric_dict in metrics_list])

print(res_dict)
df = pd.DataFrame(res_dict)
df["metrics"] = df.index
df.reset_index(inplace=True, drop=True)
cols = df.columns.tolist()
df = df[cols[-1:] + cols[:-1]]
df = df.round(decimals=2)
# df.to_clipboard()

print(df)

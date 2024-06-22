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
from scripts.custom_functions.general_for_db_speed_test import path_join
import decimal
from settings_plot import *

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 20)

mean_kge = pickle.load(open(path_join([Path.cwd().parent.parent, "df_best_runs.pkl"]), "rb"))
kge = pickle.load(open(path_join([Path.cwd().parent.parent, "df_best_runs_all_kge.pkl"]), "rb"))

kge = kge.T
kge["BS"] = kge.index.str.split("/").str[-3].str.split().str[-1].astype(int)
kge["model"] = kge.index.str.split("/").str[-6]
kge["modelBS"] = kge.model + " BS=" + kge.BS.astype(str)
# Filter
kge = kge[(kge.BS == 256) & (kge.model.str.contains("ESF"))]
kge.set_index("modelBS", inplace=True)

kge.drop(["BS", "model"], inplace=True, axis=1)

print(kge)
cnn = kge.loc["CNN +ESF BS=256"]
lstm = kge.loc["LSTM +ESF BS=256"]
gru = kge.loc["GRU +ESF BS=256"]

cnn_lstm = cnn - lstm
cnn_gru = cnn - gru
lstm_gru = lstm - gru
#cnn_lstm.plot()
#cnn_gru.plot()
lstm_gru.plot()
plt.show()
res_cnn_lstm = cnn_lstm[cnn_lstm < 0].count()
res_cnn_gru = cnn_gru[cnn_gru < 0].count()
res_lstm_gru = lstm_gru[lstm_gru < 0].count()
print(res_cnn_lstm, res_cnn_gru, res_lstm_gru)

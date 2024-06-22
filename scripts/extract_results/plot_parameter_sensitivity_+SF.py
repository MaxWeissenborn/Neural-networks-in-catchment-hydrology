import pandas as pd
from pathlib import Path
from os import listdir
import pickle
import itertools
import numpy as np
import matplotlib.ticker as ticker
from settings_plot import *


def center_ylabels(ax1, ax2):
    pos2 = ax2.get_position()
    right = pos2.bounds[0]

    pos1 = ax1.get_position()
    left = pos1.bounds[0] + pos1.bounds[2]

    offset = ((right - left) / pos2.bounds[2]) * -0.5

    for yt in ax2.get_yticklabels():
        yt.set_position((offset, yt.get_position()[1]))
        yt.set_ha('center')

        plt.setp(ax2.yaxis.get_major_ticks(), pad=0)


plot_all = False  # whether to create plots for all models. Decides which xlim is set, based on maximum value
version = 3
if plot_all:
    modelNames = ["LSTM +ESF", "CNN +ESF", "GRU +ESF"]
    bs_list = [256, 2048]
else:
    modelNames = ["LSTM +ESF", "GRU +ESF", "CNN +ESF"]
    bs_list = [256]

# modelNames = ["GRU -SF", "LSTM -SF", "CNN -SF"]


root = Path.cwd().parents[1]

for value in itertools.product(modelNames, bs_list):
    modelName = value[0]
    bs = value[1]
    ext = modelName.split(" ")[-1]
    print(modelName, bs)
    print("####################")

    input_path = root.joinpath(*["results", "sensitivity", "testing", modelName, "bs=%s" % bs])

    files = listdir(input_path)
    data = {}
    for file in files:
        file = str(file)
        if file.startswith('Q'):
            with open(input_path.joinpath(file), "rb") as f:
                q = pickle.load(f)
                if "benchmark" in file:
                    benchmark = q
                else:
                    name = file.split("-")[-1].split(".")[0]
                    data[name] = q

    df = pd.DataFrame(data.items(), columns=["Features", "Discharge"])
    df["change"] = (df.Discharge / benchmark - 1) * 100
    df.sort_values(by=['change'], ascending=False, inplace=True, ignore_index=True)
    df["group"] = bs

    if ext == "-SF":

        # second dataset
        #############################
        bs_ = 2048
        if bs == bs_:
            continue
        #############################
        input_path = root.joinpath(*["results", "sensitivity", "testing", modelName, "bs=%s" % bs_])

        files = listdir(input_path)
        data = {}
        for file in files:
            file = str(file)
            if file.startswith('Q'):
                with open(input_path.joinpath(file), "rb") as f:
                    q = pickle.load(f)
                    if "benchmark" in file:
                        benchmark = q
                    else:
                        name = file.split("-")[-1].split(".")[0]
                        data[name] = q

        df_join = pd.DataFrame(data.items(), columns=["Features", "Discharge"])
        df_join["change"] = (df_join.Discharge / benchmark - 1) * 100
        df_join.sort_values(by=['change'], ascending=False, inplace=True, ignore_index=True)
        df_join["group"] = bs_
        df = pd.concat([df, df_join], axis=0, ignore_index=True)

    ####################################

    encoding_dict = {'gesteinsart_huek250': {0: 'igneous',
                                             1: 'sedimentary'},
                     'soil_texture_boart_1000': {0: 'loam, sandy loam',
                                                 1: 'silt loam',
                                                 2: 'sandy loam',
                                                 3: 'Schlufftone (ut)',  # is not included within my data
                                                 4: 'silty clay'},
                     'durchlässigkeit_huek250': {0: 'low permeable (>1E-7 - 1E-5)',
                                                 1: 'low to very low permeable (<1E-5)',
                                                 2: 'moderate permeable  (>1E-5 - 1E-4)',
                                                 3: 'moderate to low permeable (>1E-6 - 1E-4)',
                                                 4: 'strong permeable (>1E-4 - 1E-3)',
                                                 5: 'strong to moderate permeable (>1E-5 - 1E-3)',
                                                 6: 'very low permeable (>1E-9 - 1E-7)',
                                                 7: 'very strong permeable'},
                     'dominating_soil_type_bk500': {0: 'Dystric Cambisols',
                                                    1: 'Eutric Cambisols',
                                                    2: 'Eutric Cambisols, Stagnic Gleysols',
                                                    3: 'Haplic Luvisols, Eutric Podzoluvisols, Stagnic Gleysols',
                                                    4: 'Spodic Cambisols'},
                     'land_use_corine': {0: 'Agriculture',
                                         1: 'Grassland',
                                         2: 'Forest'}}

    ##############################################################

    # reformat_features = [r'average precipitation [$\bf{mm}$]',
    #                      r'precipitation [$\bf{mm}$]',
    #                      'runoff_ratio [/]',
    #                      'dominating_soil_type_bk500=7',
    #                      'soil_texture_boart_1000=2',
    #                      'soil_texture_boart_1000=4',
    #                      'dominating_soil_type_bk500=6',
    #                      'soil_texture_boart_1000=1',
    #                      'land_use_corine=1',
    #                      'dominating_soil_type_bk500=5',
    #                      'durchlässigkeit_huek250=3',
    #                      'soil_texture_boart_1000=0',
    #                      'durchlässigkeit_huek250=2',
    #                      'land_use_corine=0',
    #                      'dominating_soil_type_bk500=0',
    #                      'average slope [/]',
    #                      'dominating_soil_type_bk500=4',
    #                      r'evapotranspiration [$\bf{mm}$]',
    #                      r'catchment size [$\bf{m^2}$]',
    #                      'elongation_ratio [/]',
    #                      'durchlässigkeit_huek250=5',
    #                      'durchlässigkeit_huek250=1',
    #                      'gesteinsart_huek250=1',
    #                      'soil temperature [°C]',
    #                      'dominating_soil_type_bk500=1',
    #                      'soil depth [m]',
    #                      'dominating_soil_type_bk500=3',
    #                      'dominating_soil_type_bk500=2',
    #                      'land_use_corine=2',
    #                      'durchlässigkeit_huek250=7',
    #                      'durchlässigkeit_huek250=0',
    #                      'gesteinsart_huek250=0',
    #                      r'average evapotranspiration [$\bf{mm}$]']
    #
    # # run only once to create a matching name dict
    # name_mapping = zip(df.Features, reformat_features)
    # for z in name_mapping:
    #     print('"%s": "%s",' % (z[0], z[1]))

    reformat_features = {"prec_mean": r"average precipitation [$\bf{mm}$]",
                         "prec_mm": r"Precipitation [$\bf{mm}$]",
                         "soil_texture_boart_1000=2": "soil_texture_boart_1000=2",
                         "soil_texture_boart_1000=4": "soil_texture_boart_1000=4",
                         "soil_texture_boart_1000=1": "soil_texture_boart_1000=1",
                         "land_use_corine=1": "land_use_corine=1",
                         "durchlässigkeit_huek250=3": "durchlässigkeit_huek250=3",
                         "soil_texture_boart_1000=0": "soil_texture_boart_1000=0",
                         "durchlässigkeit_huek250=2": "durchlässigkeit_huek250=2",
                         "land_use_corine=0": "land_use_corine=0",
                         "dominating_soil_type_bk500=0": "dominating_soil_type_bk500=0",
                         "slope_mean_dem_40": "average slope [/]",
                         "dominating_soil_type_bk500=4": "dominating_soil_type_bk500=4",
                         "et_mm": r"Evapotranspiration [$\bf{mm}$]",
                         "area_m2_watershed": r"catchment size [$\bf{m^2}$]",
                         "elongation_ratio": "elongation ratio [/]",
                         "durchlässigkeit_huek250=5": "durchlässigkeit_huek250=5",
                         "durchlässigkeit_huek250=1": "durchlässigkeit_huek250=1",
                         "gesteinsart_huek250=1": "gesteinsart_huek250=1",
                         "soil_temp": "Soil temperature [°C]",
                         "dominating_soil_type_bk500=1": "dominating_soil_type_bk500=1",
                         "greundigkeit_physgru_1000": "soil depth [m]",
                         "dominating_soil_type_bk500=3": "dominating_soil_type_bk500=3",
                         "dominating_soil_type_bk500=2": "dominating_soil_type_bk500=2",
                         "land_use_corine=2": "land_use_corine=2",
                         "durchlässigkeit_huek250=7": "durchlässigkeit_huek250=7",
                         "durchlässigkeit_huek250=0": "durchlässigkeit_huek250=0",
                         "gesteinsart_huek250=0": "gesteinsart_huek250=0",
                         "et_mean": r"Average evapotranspiration [$\bf{mm}$]"}

    new_reformat_features = list(df.Features)
    if ext == "+ESF":
        for item in encoding_dict:
            # print(item)
            for old_feature_name, feature in reformat_features.items():
                # if _ in new_reformat_features:
                idx = list(df.Features).index(old_feature_name)
                if feature.startswith(item):
                    if feature.startswith("land_use"):
                        new_name = "land use = %s" % encoding_dict[item][int(feature.split("=")[-1])]
                    elif feature.startswith("dominating_soil_type"):
                        new_name = "soil type = %s" % encoding_dict[item][int(feature.split("=")[-1])]
                    elif feature.startswith("durchlässigkeit_huek250"):
                        new_name = "permeability = %s" % encoding_dict[item][int(feature.split("=")[-1])]
                        # new_name = "permeability [$ms^{-1}$] = %s" % encoding_dict[item][int(feature.split("=")[-1])]
                    elif feature.startswith("soil_texture_boart_1000"):
                        new_name = "soil texture = %s" % encoding_dict[item][int(feature.split("=")[-1])]
                    elif feature.startswith("gesteinsart_huek250"):
                        new_name = "geology type = %s" % encoding_dict[item][int(feature.split("=")[-1])]
                    else:
                        new_name = feature.split("=")[0] + "=%s" % encoding_dict[item][int(feature.split("=")[-1])]
                    new_reformat_features[idx] = new_name
                else:
                    new_reformat_features[idx] = r"%s" % feature

            reformat_features = {k: v for k, v in reformat_features.items() if not v.startswith(item)}
    else:
        new_reformat_features = [reformat_features[i] for i in new_reformat_features if i in reformat_features]

    # print(new_reformat_features)

    df.Features = new_reformat_features

    # _ = []
    # for f in df.Features:
    #     if "=" in f:
    #         f = f.split("=")[0]
    #         if f not in _:
    #             _.append(f)
    #             print(f + " &       &       \\\\")
    #     else:
    #         print(f + " &       &       \\\\")
    #
    #

    df.Features = df.Features.str.replace(r"\(.*\)", "", regex=True)

    # df.Features = np.where(df.Features == "soil type = Haplic Luvisols, Eutric Podzoluvisols, Stagnic Gleysols",
    #                        "soil type = Haplic luvisols, eutric podzoluvisols,\nstagnic gleysols", df.Features)

    df_pos = df.copy()
    df_pos["change"] = np.where(df_pos.change >= 0, df_pos.change, 0)

    df_neg = df.copy()
    df_neg["change"] = np.where(df_neg.change < 0, df_neg.change, 0)

    df_list = [df_neg, df_pos]

    # Draw Plot
    if ext == "+ESF":
        if plot_all:
            fig, axes = plt.subplots(1, 2, figsize=(20, 12),
                                     gridspec_kw={'width_ratios': [1, 2.46]})
        else:
            # for Thesis
            fig, axes = plt.subplots(1, 2, figsize=(20, 12), gridspec_kw={'width_ratios': [1, 1.45]})

    else:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8),
                                 gridspec_kw={'width_ratios': [1, 2.08]})

    fig.subplots_adjust(wspace=0.5)

    category_to_color = {
        'soil texture': '#DE8F05',
        'soil type': '#DE8F05',  # yellow
        'land use': '#108010',  # green
        'permeability': 'black',  # lightblue #2EABB8
        'geology type': "black",  # '#CC78BC',
        'default': '#9D4139'  # brown
    }

    df['color'] = df['Features'].apply(lambda x: category_to_color.get(x.split(' = ')[0], category_to_color['default']))

    for row in df.itertuples():
        if row.color == category_to_color["default"]:
            # print(row.Features.lower())
            if "precipitation" in row.Features.lower():
                df.loc[row.Index, 'color'] = "#176D9C"

            elif "evapotranspiration" in row.Features.lower():
                df.loc[row.Index, 'color'] = "#176D9C"

            elif "soil depth" in row.Features.lower():
                df.loc[row.Index, 'color'] = "#DE8F05"

            elif "soil temperature" in row.Features.lower():
                df.loc[row.Index, 'color'] = "#DE8F05"

            # add daily or yearly
            if row.Features == r"Precipitation [$\bf{mm}$]":
                df.loc[row.Index, 'Features'] = r"Daily sum precipitation [$\bf{mm}$]"

            if row.Features == r"average precipitation [$\bf{mm}$]":
                df.loc[row.Index, 'Features'] = r"Yearly average precipitation [$\bf{mm}$]"

            if row.Features == r"Evapotranspiration [$\bf{mm}$]":
                df.loc[row.Index, 'Features'] = r"Daily sum evapotranspiration [$\bf{mm}$]"

            if row.Features == r"Average evapotranspiration [$\bf{mm}$]":
                df.loc[row.Index, 'Features'] = r"Yearly average evapotranspiration [$\bf{mm}$]"

            if row.Features == r"Soil temperature [°C]":
                df.loc[row.Index, 'Features'] = r"Daily average soil temperature [°C]"


    # rename for the thousand time
    # conditional y axis label formatting
    category_features = ['geology type',
                         'soil texture',
                         'permeability',
                         'soil type',
                         "land use"]

    non_static_features = ["Daily sum precipitation", "Daily sum evapotranspiration", "Daily average soil temperature"]

    for row in df.itertuples():
        f = row.Features
        for elem in category_features:
            if f.startswith(elem):
                df.loc[row.Index, 'Features'] = f.split(" = ")[-1]

    ax_list = []

    for i in range(2):

        df.Features = df.Features.str.capitalize()
        if ext == "+ESF":
            sub_plot = sns.barplot(x='change', y=df.Features, data=df_list[i], ax=axes[i],
                                   orient='h', palette=list(df['color']), saturation=1)

        else:  # -SF
            sub_plot = sns.barplot(x='change', y='Features', data=df_list[i], hue="group", ax=axes[i], saturation=1)

        sub_plot.set(ylabel=None)

        if i == 0:

            if ext == "+ESF":
                sub_plot.set_xlim(-11)
            else:

                for i in sub_plot.containers:
                    sub_plot.bar_label(i, fmt='%.1f', label_type="center", size=14, weight="bold", color="white")

                sub_plot.set_xlim(-11)
                plt.setp(sub_plot.get_legend().get_title(), fontsize='19')
                sns.move_legend(sub_plot, "center", bbox_to_anchor=(0.275, 0.888),  # ncol=2,
                                shadow=False, title="Batch size",
                                frameon=~False)

            sub_plot.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
            sns.despine(left=True, bottom=True, right=False, ax=sub_plot)
            sub_plot.set_xlabel("Decrease of discharge in %", labelpad=30)

        #     sub_plot.invert_xaxis()
        #     sub_plot.yaxis.tick_right()
        else:
            sub_plot.legend([], [], frameon=False)
            if ext == "+ESF":
                if plot_all:
                    sub_plot.set_xlim(xmin=0, xmax=16)
                else:
                    # this forThesis only
                    sub_plot.set_xlim(xmin=0, xmax=16)

                # set custom legend
                custom_legend = [(category_to_color["default"], "Catchment characteristics"), (category_to_color["land use"], 'Land use'),
                                 (category_to_color["geology type"], 'Geology'), ("#176D9C", "Meteorology"),
                                 # ('#2EABB8', 'Permeability'),
                                 # ('#882255', 'Soil texture'),
                                 (category_to_color["soil texture"], 'Soil')]
                patches = [matplotlib.patches.Patch(color=elem[0], label=elem[1]) for elem in custom_legend]
                patches.append(matplotlib.patches.Patch(None, label="Metric features", fill=False, linewidth=0))
                patches.append(matplotlib.patches.Patch(None, label="Daily forcing data", fill=False, linewidth=0))
                patches.append(matplotlib.patches.Patch(None, label="Categorical features", fill=False, linewidth=0))

                if plot_all:
                    sub_plot.legend(handles=patches, loc="center", frameon=False, bbox_to_anchor=(0.562, 0.249))
                    frame_width = 8.8
                    frame_x = 4.6
                    text_x = 5
                else:
                    sub_plot.legend(handles=patches, loc="center", frameon=False, bbox_to_anchor=(0.625, 0.249))
                    frame_width = 10
                    frame_x = 5
                    text_x = 5.2
                rect = matplotlib.patches.FancyBboxPatch((frame_x, 16.5), width=frame_width, height=9.5, linewidth=.4,
                                                         edgecolor='grey', facecolor='white',
                                                         boxstyle="round, pad=0.0,rounding_size=0.2")
                sub_plot.add_patch(rect)
                sub_plot.annotate("Text", xy=(text_x, 23.1), weight='bold', fontsize=16, color="black")
                sub_plot.annotate("Text", xy=(text_x, 24.21), weight='bold', fontsize=16, color="#71706E")
                sub_plot.annotate("Text", xy=(text_x, 25.29), fontsize=16, color="black")
            else:

                for i in sub_plot.containers:
                    sub_plot.bar_label(i, fmt='%.1f', label_type="center", size=14, weight="bold", color="white")

                sub_plot.set_xlim(xmin=0, xmax=22.9)
                # plt.legend(loc='lower right', title="Batch size")

            sns.despine(left=False, bottom=True, right=True, ax=sub_plot)
            sub_plot.set_xlabel("Increase of discharge in %", labelpad=30)

        # highlight all numeric features
        print(df.to_string())

        metric_features = non_static_features + ["Yearly average evapotranspiration ", "Runoff ratio [/]",
                                                 "Yearly average precipitation",
                                                 "Catchment size", "Elongation ratio [/]",
                                                 "Soil depth [m]", "Average slope [/]"]
        for idx, row in df.iterrows():

            # for elem in category_features:
            #     if row["Features"].lower().startswith(elem):
            #         print(sub_plot.get_yticklabels())
            #
            feature = row["Features"]  # .split("=")[0]

            if feature.startswith(tuple(metric_features)):

                # make all bold
                if ext == "+ESF":
                    sub_plot.get_yticklabels()[idx].set_fontweight("bold")
                    if feature.startswith(tuple(non_static_features)):
                        # make all non-static also green
                        sub_plot.get_yticklabels()[idx].set_color("#71706E")
                else:
                    if idx < len(df) / 2:
                        sub_plot.get_yticklabels()[idx].set_fontweight("bold")

        if ext == "+ESF":
            # set every fourth tick
            n = 4
            # ax.xaxis.set_major_locator(MultipleLocator(n))
            sub_plot.grid(alpha=0.4, color="black", linestyle=":")
            # remove unwanted gridlines on the y-axis
            y_grd_lines = sub_plot.get_ygridlines()
            [grd_line.set_visible(False) for i, grd_line in enumerate(y_grd_lines) if i % n]

        sub_plot.xaxis.set_major_locator(ticker.MultipleLocator(2))

        ax_list.append(sub_plot)

    for i in range(2):
        for j in axes[i].containers:
            for k in j:
                width = k.get_width()

                if width != 0.0:
                    if abs(width) <= 0.1:
                        if i == 0:
                            axes[i].text(width - 0.1, k.get_y() + k.get_height() - 0.2, width.round(2), ha='right',
                                         size=14, weight="bold", color="black", )
                        else:
                            axes[i].text(width + 0.1, k.get_y() + k.get_height() - 0.2, width.round(2), ha='left',
                                         size=14,
                                         weight="bold", color="black", )
                    else:
                        if i == 0:
                            axes[i].text(width - 0.1, k.get_y() + k.get_height() - 0.2, width.round(1), ha='right',
                                         size=14, weight="bold", color="black", )
                        else:
                            axes[i].text(width + 0.1, k.get_y() + k.get_height() - 0.2, width.round(1), ha='left',
                                         size=14,
                                         weight="bold", color="black", )

    plt.tight_layout(pad=2.5)
    center_ylabels(*ax_list)

    plt.show()

    # df["category"] = np.where(df.color == "#DE8F05", "soil",
    #                                np.where(df.color == "#108010", "land use",
    #                                         np.where(df.color == "black", "geology",
    #                                                  np.where(df.color == "#176D9C", "meteorology", "catchment characteristics"))))
    #
    # df.drop("Discharge", inplace=True, axis = 1)
    # # df.drop("group", inplace=True, axis=1)
    # df.drop("color", inplace=True, axis=1)
    # df = df.round(2)
    # df.to_csv("senitivity_%s_%s_data.csv" % (modelName, bs), sep=",", index=True)
    # print(df.to_string())

    # continue

    if ext == "+ESF":
        fig.savefig(root.joinpath(*["results", "Sensitivity-%s-bs=%s_v%s.pdf" % (modelName, bs, version)]),
                    format="pdf", bbox_inches="tight")
    else:
        fig.savefig(root.joinpath(*["results", "Sensitivity-%s_v%s.pdf" % (modelName, version)]),
                    format="pdf", bbox_inches="tight")
    plt.close(fig)

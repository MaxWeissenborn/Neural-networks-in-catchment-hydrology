import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


def load_data(filename):
    # load in data
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    df = pd.read_csv(Path.joinpath(input_path, filename), sep=";", date_parser=dateparse, index_col="date")
    return df


def check_for_nan(data):
    col_names = list(data[data.columns[data.isna().any()]])
    return col_names


def clip_df(data):
    # clip rows
    data = data[pd.Timestamp(start):pd.Timestamp(end)]
    # clip cols
    data = data.loc[:, data.columns.isin(name_list)]
    return data


def remove_na(data):
    # creating dataset with no missing values
    print(len(data.columns[data.isna().any()].tolist()), "Columns have missing values and will be removed.")
    data = data.drop(data.columns[data.isna().any()], axis=1)
    print("data has now the following shape: %s\n" % str(data.shape))
    return data


def save(data, filename):
    # save new data file
    name = "_".join(filename.split("_")[:2] + [start_y, end_y]) + ".csv"
    data.to_csv(Path.joinpath(output_path, 'delete_NO NA - TESTING NOT FOR TRAIN - ' + name), encoding='utf-8')

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 20)

cwd = Path.cwd()
if cwd.stem == "custom_functions":
    base = cwd.resolve().parent.parent
else:
    base = cwd

input_path = Path.joinpath(base, "preprocessing", "raw data")
output_path = Path.joinpath(base, "preprocessing", "output")

for add_static_features in [True, False]:
    if add_static_features:
        prefix = "+ESF"
    else:
        prefix = "-SF"

    filenames = ["et_mm_1991_2018_corrected.csv",
                 "prec_mm_1991_2018.csv",
                 "soil_temp_C_1991_2018.csv",
                 "dis_mm_1991_2018.csv"]

    name_list = []
    data_dict = {}

    for f in filenames:
        temp_df = load_data(f)
        # get gauge_ids with missing data
        name_list = name_list + check_for_nan(temp_df)
        data_dict[f] = temp_df

    # discharge data is the only dataset with missing values
    df = data_dict["dis_mm_1991_2018.csv"]
    # filter dataframe to gauge_ids with missing values
    df = df[df.columns[df.isna().any()]]
    # filter to select only gauge_ids with less than 1000 days of missing data
    df = df[df.columns[df.isna().sum() < 1000]]
    # create a new column with an unique id per row and drop all missing data
    _ = df.assign(id=df.reset_index().index).dropna()
    # find longest steady periode
    idx = _.id.groupby(_.id.diff().ne(1).cumsum()).transform('count')
    # show time index range
    print(_[idx == idx.max()].index)

    start = "1997-01-01"
    end = "2002-12-31"
    start_y = start.split("-")[0]
    end_y = end.split("-")[0]

    river_data = {}
    for k, v in data_dict.items():
        print(k, "data has the following shape:", v.shape)
        temp_dict = clip_df(v)
        df = remove_na(temp_dict)
        data_dict[k] = df
        save(df, k)
        river_data["_".join(k.split("_")[:2])] = df

    df_catchment = pd.read_csv(Path.joinpath(output_path, "delete_NO NA cleaned_catchment_attributes_num.csv"), sep=",")
    catchments = df_catchment["gauge_id"].unique()
    # make gauge_id new index of catchment dataframe
    df_catchment.set_index('gauge_id', inplace=True)

    # create dataframe for every catchment containing all attributes of filenames as columns
    df_dict = {}
    for c in catchments:
        str_c = str(c)
        temp_df = pd.DataFrame()
        for k, v in river_data.items():
            if str_c in river_data[k].columns:
                temp_df[k] = river_data[k][str_c]

        if len(temp_df.columns) > 3:
            df_dict[c] = temp_df

    final_data_dict = {}
    if add_static_features:
        """
        get all column names of catchment data set. loop over river data sets to get every river id and the corresponding
        river data. Fetch the data from catchment dataset which is one row for every river. Repeat this line n times 
        for every day. Concat it with river data to gain one data set for every river with all information. Store every 
        data set in a dictionary with k = gauge_id and v = dataframe
        """

        new_cols = list(df_catchment.columns)
        for k, v in df_dict.items():
            temp_df = pd.DataFrame(np.repeat(df_catchment.loc[k].values[np.newaxis, ...], v.shape[0], axis=0),
                                   columns=new_cols)
            temp_df_concat = pd.concat([v, temp_df.set_index(v.index)], axis=1, ignore_index=False)
            final_data_dict[k] = temp_df_concat

    if len(final_data_dict) == 0:
        final_data_dict = df_dict

    # show 1 example river
    sample = list(final_data_dict.keys())[0]
    print(final_data_dict[sample])

    print("There are %s rivers in this data set" % len(final_data_dict))
    shape_ = final_data_dict[sample].shape
    print("The data set has Information for %s days or %s years" % (shape_[0], int(shape_[0] / 365)))
    out_name = "%s - NO NA - TESTING NOT FOR TRAIN - Complete River Data - %s - %s.pkl" % (prefix, start_y, end_y)
    pickle.dump(final_data_dict, open(Path.joinpath(output_path, out_name), "wb"))

    #################################################################################
    # Combine all rivers to one dataset with a new column representing the river id #
    #################################################################################

    # create new column in every dataframe with gauge_id as value
    for k, df in final_data_dict.items():
        df["gauge_id"] = k

    # concat all dataframes
    df_concat = pd.concat(tuple(final_data_dict[i] for i in [*final_data_dict]))
    out_name = "%s - NO NA - TESTING NOT FOR TRAIN - Complete River Data as Dataframe - %s - %s.pkl" % (
        prefix, start_y, end_y)
    pickle.dump(df_concat, open(Path.joinpath(output_path, out_name), "wb"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import h5py
import pickle
from pathlib import Path


def path_join(path_list):
    """ takes a list of strings and add these to the "current working directory" -1 step back"""
    base = Path.cwd()  # .parents[0]

    if str(base) in str(path_list[0]):
        # extend custom path
        p = Path.joinpath(Path(*path_list))
    else:
        # extend basepath
        p = base.joinpath(*path_list)

    return p


def create_h5_db(t, d, out_file, train_set=None, validation_set=None, test_set=None):
    """
    create_h5_db

    Parameters:
    t (int): Timesteps
    d (int): Number of features
    out_file (str): Path to output h5 file
    train_set (tuple, optional): Tuple containing X_train and y_train arrays
    validation_set (tuple, optional): Tuple containing X_val and y_val arrays
    test_set (tuple, optional): Tuple containing X_test and y_test arrays

    Functionality:
    Creates a h5 database file with train, validation and test datasets.
    The X datasets contain the features with shape (num_samples, timesteps, num_features).
    The Y datasets contain the targets with shape (num_samples, 1).
    The data is stored compressed using lzf compression.
    Chunks are used to allow efficient reads.
    """

    with h5py.File(out_file, 'w') as f:
        compression = 'lzf'  #'gzip'
        if train_set is not None:
            x_train, y_train = train_set
            f.create_dataset(
                'X_train',
                shape=(x_train.shape[0], t, d),
                #maxshape=(None, t, d),
                chunks=True,
                dtype=np.float32,
                compression=compression,
                data=x_train)
            f.create_dataset(
                'Y_train',
                shape=(y_train.shape[0], 1),
                #maxshape=(None, 1),
                chunks=True,
                dtype=np.float32,
                compression=compression,
                data=y_train)
        if validation_set is not None:
            x_val, y_val = validation_set
            f.create_dataset(
                'X_val',
                shape=(x_val.shape[0], t, d),
                #maxshape=(None, t, d),
                chunks=True,
                dtype=np.float32,
                compression=compression,
                data=x_val)
            f.create_dataset(
                'Y_val',
                shape=(y_val.shape[0], 1),
                #maxshape=(None, 1),
                chunks=True,
                dtype=np.float32,
                compression=compression,
                data=y_val)
        if test_set is not None:
            x_test, y_test = test_set
            f.create_dataset(
                'X_test',
                shape=(x_test.shape[0], t, d),
                #maxshape=(None, t, d),
                chunks=True,
                dtype=np.float32,
                compression=compression,
                data=x_test)
            f.create_dataset(
                'Y_test',
                shape=(y_test.shape[0], 1),
                #maxshape=(None, 1),
                chunks=True,
                dtype=np.float32,
                compression=compression,
                data=y_test)

        f.flush()


def split_to_train_and_validation(df):
    n_river = len(df) / len(df["gauge_id"].unique())
    Ntrain = n_river * 4 // 5  # using ca 80 % of the data to train
    Ntest = n_river - Ntrain
    train = df.groupby("gauge_id").head(Ntrain)
    test = df.groupby("gauge_id").tail(Ntest)

    return train, test


def normalize(train, validate, test, gs, suffix):
    """Normalizes all columns of a dataframe

    Parameters:
      dataframe (dataframe):
      T (int): Number of time stamps
      useMinMaxScaler (bool): Wether to use the MinMaxScaler. Default is Standartscaler.

    Returns:
      array:The normalized dataset.
      :param test:
      :param validate:
      :param train:
    """

    scaler_feature = MinMaxScaler(feature_range=(0, 1))
    scaler_target = MinMaxScaler(feature_range=(0, 1))

    train_gauge_id = train[["gauge_id"]].reset_index(drop=True)
    train = train.drop("gauge_id", axis=1, inplace=False)
    val_gauge_id = validate[["gauge_id"]].reset_index(drop=True)
    validate = validate.drop("gauge_id", axis=1, inplace=False)

    train_feature_norm = scaler_feature.fit_transform(train.iloc[:, :-1])
    train_target_norm = scaler_target.fit_transform(train.iloc[:, [-1]])

    val_feature_norm = scaler_feature.transform(validate.iloc[:, :-1])
    val_target_norm = scaler_target.transform(validate.iloc[:, [-1]])

    # recreate data frame
    col_names = train.columns
    df_train_norm = pd.DataFrame(data=np.column_stack((train_feature_norm, train_target_norm)), columns=col_names)
    df_train_norm = pd.concat([df_train_norm, train_gauge_id], axis=1)

    df_val_norm = pd.DataFrame(data=np.column_stack((val_feature_norm, val_target_norm)), columns=col_names)
    df_val_norm = pd.concat([df_val_norm, val_gauge_id], axis=1)

    # save scaler
    pickle.dump(scaler_feature, open(path_join(["scaler - %s - feature 80-20 split.pkl" % suffix]), "wb"))
    pickle.dump(scaler_target, open(path_join(["scaler - %s - target 80-20 split.pkl" % suffix]), "wb"))

    if test is not None:

        test_gauge_id = test[["gauge_id"]].reset_index(drop=True)
        test = test.drop("gauge_id", axis=1, inplace=False)

        test_feature_norm = scaler_feature.transform(test.iloc[:, :-1])
        test_target_norm = scaler_target.transform(test.iloc[:, [-1]])
        df_test_norm = pd.DataFrame(data=np.column_stack((test_feature_norm, test_target_norm)), columns=col_names)
        df_test_norm = pd.concat([df_test_norm, test_gauge_id], axis=1)
    else:
        df_test_norm = None

    return df_train_norm, df_val_norm, df_test_norm


def data_converter(dfs_dict, t, d, out_file):
    """ Creating an array with T x D where T is number of time stamps and D is the number of features
    """
    result = {}
    for df_name, df in dfs_dict.items():
        df_name = df_name.split("_")[0]
        x = []
        y = []
        for name, group in df.groupby('gauge_id'):

            sub_df = group
            sub_df.drop(labels=["gauge_id"], axis=1, inplace=True)
            sub_df_values = sub_df.values

            for i in range(len(sub_df_values) - t):
                x_ = sub_df_values[i:i + t, :-1]
                x.append(x_)
                y_ = sub_df_values[i + t, -1]
                y.append(y_)

        x = np.array(x).reshape(-1, t, d).astype('float32')  # Now the data should be N-T+1 x T x D
        y = np.array(y).reshape(-1, 1).astype('float32')

        result[df_name] = [x, y]

    if "train" in result:
        create_h5_db(t, d, out_file, train_set=result["train"], validation_set=result["val"])
    else:
        create_h5_db(t, d, out_file, test_set=result["test"])

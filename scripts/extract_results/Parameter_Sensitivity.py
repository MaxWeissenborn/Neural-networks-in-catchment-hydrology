import os.path
from pathlib import Path
import spotpy
import pandas as pd
from scripts.custom_functions.general_for_db_speed_test import path_join, load_yaml_file, make_dir, clean_up
import h5py
import pickle
from types import SimpleNamespace
import glob
import numpy as np
from scripts.custom_functions.database_preparation import data_converter
from tensorflow.keras.models import load_model
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
import yaml


def load_scaler(fix):

    scaler_feature = pickle.load(open(glob.glob("scaler*%s*feature*" % fix)[0], "rb"))
    scaler_target = pickle.load(open(glob.glob("scaler*%s*target*" % fix)[0], "rb"))
    return scaler_feature, scaler_target


def normalize(df, fix_):
    scaler_feature, scaler_target = load_scaler(fix_)
    test = df
    test_gauge_id = test[["gauge_id"]].reset_index(drop=True)
    test = test.drop("gauge_id", axis=1, inplace=False)

    col_names = test.columns

    test_feature_norm = scaler_feature.transform(test.iloc[:, :-1])
    test_target_norm = scaler_target.transform(test.iloc[:, [-1]])
    df_test_norm = pd.DataFrame(data=np.column_stack((test_feature_norm, test_target_norm)), columns=col_names)
    df_test_norm = pd.concat([df_test_norm, test_gauge_id], axis=1)

    return df_test_norm


def process(df, feature_name, model_name, bs, t, d, fix, db_path, trained_model_path, gs):

    progress_dict_file = "sensitivity_progress_dict.pkl"
    if os.path.isfile(progress_dict_file):
        progress_dict = pickle.load(open(progress_dict_file, "rb"))
        # show progress
        for k, v in progress_dict.items():
            for k1, v1 in v.items():
                print(k, len(v1))

    else:
        progress_dict = {}

    if model_name in progress_dict:
        if bs in progress_dict[model_name].keys():
            if feature_name in progress_dict[model_name][bs]:
                return
        else:
            progress_dict[model_name][bs] = []
    else:
        progress_dict[model_name] = {bs: []}

    db_name = f"{model_name}_{feature_name}_bs={bs}.hdf5"
    db = db_path.joinpath(db_name)
    df_norm = normalize(df, fix)
    data_converter({"test_norm": df_norm}, t, d, db)
    test_model(db, feature_name, model_name, fix, trained_model_path, gs, bs)
    progress_dict[model_name][bs].append(feature_name)

    pickle.dump(progress_dict, open(progress_dict_file, "wb"))


def test_model(db, feature_name, model_name, fix, trained_model_path, gs, bs):

    #print(db)

    output_path = path_join([Path(db).parent.parent.parent, "testing", model_name, "bs=%s" % bs])
    #print(output_path)
    make_dir(output_path)
    if not os.path.isfile(path_join([output_path, "Q_mean-%s.pkl" % feature_name])):
        #print(model_name)

        model = load_model(trained_model_path, compile=False)
        model.compile(loss=CustomLoss(), optimizer=tf.keras.optimizers.Adam(learning_rate=gs.lr))

        with h5py.File(db, 'r') as data:
            x = data["X_test"][...]
            y = data["Y_test"][...].flatten()

            prediction = model.predict(x, verbose=2)[:, 0].flatten()  # flatten was needed now

        n = gs.test_samples
        split = int(len(y) / n)

        # unscale data
        _, scaler_target = load_scaler(fix)
        # create empty table with y x n fields
        y_fake_dataset = np.zeros(shape=(len(y), n))
        p_fake_dataset = np.zeros(shape=(len(prediction), n))
        # put the predicted values in the right field
        y_fake_dataset[:, -1] = y
        p_fake_dataset[:, -1] = prediction
        # inverse transform and then select the right field
        y_unscaled = scaler_target.inverse_transform(y_fake_dataset)[:, -1]
        p_unscaled = scaler_target.inverse_transform(p_fake_dataset)[:, -1]

        pickle.dump(y_unscaled, open(path_join([output_path, "observation_unscaled-%s.pkl" % feature_name]), "wb"))
        pickle.dump(p_unscaled, open(path_join([output_path, "prediction_unscaled-%s.pkl" % feature_name]), "wb"))

        kge_list = []
        for i in range(n):

            y_sample = y_unscaled[i * split:(i + 1) * split]
            p_sample = p_unscaled[i * split:(i + 1) * split]

            if len(y_sample) * n != len(y):
                print(
                    "--> Test samples have no equal sizes, check 'test_samples' within global Settings")
                break

            kge = spotpy.objectivefunctions.kge(y_sample, p_sample)
            kge_list.append(kge)

        clean_up()
        # save results
        pickle.dump(kge_list, open(path_join([output_path, "KGE_test-results-%s.pkl" % feature_name]), "wb"))

        p_sample_mean = np.mean(p_unscaled)
        print(np.mean(p_unscaled))
        pickle.dump(p_sample_mean, open(path_join([output_path, "Q_mean-%s.pkl" % feature_name]), "wb"))


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        alpha = K.std(y_pred) / K.std(y_true)
        beta = K.sum(y_pred) / K.sum(y_true)  # no need to calc mean
        r = tfp.stats.correlation(y_true, y_pred, sample_axis=None, event_axis=None)
        return K.sqrt(K.square(1 - r) + K.square(1 - alpha) + K.square(1 - beta))


def main():
    path = "../../df_best_runs.pkl"

    with open(path, "rb") as f:
        best_models = pickle.load(f)
    print(best_models.to_string())

    best_models["path"] = best_models.apply(lambda row: Path(row["path"]).parents[1] / row['model'], axis=1)
    # remove relative path (../) part
    best_models["path"] = best_models['path'].apply(lambda x: Path(*Path(x).parts[2:]))
    print(best_models.to_string())
    best_models = best_models.query("(BS == 256) & (model.str.endswith('+ESF'))") # todo remove if all models should be benchmaeked
    print(best_models.to_string())

    test_df_dict = {"+ESF": "preprocessing/output/+ESF - NO NA - TESTING NOT FOR TRAIN - Complete River Data as Dataframe - 1997 - 2002.pkl",
                    "-SF": "preprocessing/output/-SF - NO NA - TESTING NOT FOR TRAIN - Complete River Data as Dataframe - 1997 - 2002.pkl"}

    root = Path.cwd().parent.parent
    os.chdir(root)
    # load global settings
    gs = SimpleNamespace(**load_yaml_file("globalSettings.yml"))

    with open(test_df_dict["+ESF"], "rb") as f:
        df_test_ESF = pickle.load(f)
        print(test_df_dict["+ESF"])
    0/0

    with open(test_df_dict["-SF"], "rb") as f:
        df_test_SF = pickle.load(f)

    for row in best_models.itertuples():

        trained_model_path = Path(row.path)
        model_parameter_path = Path(row.path).parent
        print(model_parameter_path)
        modelName = row.model
        bs = row.BS
        fix = modelName.split(" ")[-1]

        with open(model_parameter_path.joinpath("parameter.yml"), 'r') as stream:
            model_args = yaml.safe_load(stream)["model_args"]

        t = model_args["t"]
        lr = model_args["lr"]
        print(bs)
        print(modelName)

        if fix.endswith("-SF"):  # not
            continue

        if fix == "+ESF":
            df_test = df_test_ESF.copy()
        elif fix == "-SF":
            df_test = df_test_SF.copy()
        else:
            print(f"Wrong fix: {fix}")
            continue

        d = df_test.shape[1] - 2  # - gauge_id and target
        df_test['target'] = df_test[gs.target_name]
        df_test.drop(labels=[gs.target_name], axis=1, inplace=True)

        # categorical data
        category_features = ['gesteinsart_huek250',
                             'soil_texture_boart_1000',
                             'durchlässigkeit_huek250',
                             'dominating_soil_type_bk500',
                             "land_use_corine"]

        metric_features = ['et_mm', 'prec_mm', 'soil_temp', 'area_m2_watershed',
                           'greundigkeit_physgru_1000', 'slope_mean_dem_40', 'elongation_ratio',
                           'et_mean', 'prec_mean']

        test_db_file = Path(gs.path_to_testing_df[fix])
        out_path_sensitivity = Path.joinpath(test_db_file.parent, "sensitivity")
        make_dir(out_path_sensitivity)
        # todo make the following line dynamic
        # out_name_preprocessing = "NO NA - VALIDATION NOT FOR TRAIN - Complete River Data as Dataframe - 1997 - 2002"
        test_db_name = test_db_file.stem
        #print(test_db_name)

        db_path = path_join(["results", "sensitivity", "db", "bs=%s" % bs])
        make_dir(db_path)

        # set benchmark
        tmp_df = df_test.copy()
        _ = Path.joinpath(out_path_sensitivity, test_db_name + "_bs_%s_%s_benchmark.pkl" % (bs, modelName))
        if not os.path.isfile(_):
            pickle.dump(tmp_df, open(_, "wb"))

        print("- benchmark")
        process(tmp_df, "benchmark", modelName, bs, t, d, fix, db_path, trained_model_path, gs)

        # sensitivity part
        for m_feature in metric_features:
            if m_feature not in df_test.columns:
                print("ERROR: %s not found in df" % m_feature)
                continue
            print("- %s" % m_feature)
            tmp_df = df_test.copy()
            tmp_df[m_feature] = tmp_df[m_feature] * 1.1
            _ = Path.joinpath(out_path_sensitivity, test_db_name + " _%s.pkl" % m_feature)
            if not os.path.isfile(_):
                pickle.dump(tmp_df, open(_, "wb"))
            process(tmp_df, m_feature, modelName, bs, t, d, fix, db_path, trained_model_path, gs)

        for c_features in category_features:
            if c_features not in df_test.columns:
                print("ERROR: %s not found in df" % c_features)
                continue
            sub_features = list(map(int, df_test[c_features].unique()))
            for s_features in sub_features:
                tmp_df = df_test.copy()
                tmp_df[c_features] = s_features
                feature = "%s=%s" % (c_features, s_features)
                print("- %s" % feature)
                _ = Path.joinpath(out_path_sensitivity, test_db_name + " _%s.pkl" % feature)
                if not os.path.isfile(_):
                    pickle.dump(tmp_df, open(_, "wb"))

                process(tmp_df, feature, modelName, bs, t, d, fix, db_path, trained_model_path, gs)


if __name__ == '__main__':
    main()

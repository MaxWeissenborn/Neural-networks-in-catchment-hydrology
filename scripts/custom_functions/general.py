import os
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import random
import gc

import spotpy.objectivefunctions

from scripts.custom_functions.spotpy_handler import *
import sys
import yaml
import glob
import inspect  # for fetching arguments of a function
import time
import re
from IPython.display import clear_output
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from scripts.custom_functions.database_preparation import *
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
from tqdm import tqdm


gpu = tf.config.list_physical_devices('GPU')
if len(gpu) > 0:
    tf.config.experimental.set_memory_growth(gpu[0], True)  # limits gpu memory


class CustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        alpha = K.std(y_pred) / K.std(y_true)
        beta = K.sum(y_pred) / K.sum(y_true)  # no need to calc mean
        r = tfp.stats.correlation(y_true, y_pred, sample_axis=None, event_axis=None)
        return K.sqrt(K.square(1 - r) + K.square(1 - alpha) + K.square(1 - beta))


class PlotLearning(tf.keras.callbacks.Callback):
    """
    Callback to plot the learning curves of the model during training.
    """

    def on_train_begin(self, logs={}):
        self.metrics = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs={}):
        # Storing metrics
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # Plotting
        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric)

            if 'val_' + metric != 'val_lr':
                if logs['val_' + metric]:
                    axs[i].plot(range(1, epoch + 2),
                                self.metrics['val_' + metric],
                                label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.show()


def timer(start, end):
    temp = end - start
    hours = temp // 3600
    temp = temp - 3600 * hours
    minutes = temp // 60
    seconds = temp - 60 * minutes
    return '%d min %d sec' % (minutes, seconds)


def load_yaml_file(file=None, silent=False):
    # load global Settings
    if file is None:
        output = "globalSettings.yml"
    else:
        output = file
    try:
        with open(path_join([output]), 'r') as stream:
            try:
                settings = yaml.safe_load(stream)
                if file is None:
                    logger.info('Global setting were imported successfully')
                    return settings

                if not silent:
                    logger.info('--> Parameter were imported successfully')

                return settings

            except yaml.YAMLError as exc:
                logger.error(exc)
                sys.exit()
    except FileNotFoundError as exc:
        logger.error(exc)
        sys.exit()


def save_to_yaml(data, path, name):
    with open(path_join([path, name]), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def unscale(y, p, n):
    scaler = pickle.load(open(glob.glob(str(path_join([Path.cwd(), "*scaler*target*"])))[0], "rb"))

    # create empty table with y x n fields
    y_fake_dataset = np.zeros(shape=(len(y), n))
    p_fake_dataset = np.zeros(shape=(len(p), n))
    # put the predicted values in the right field
    y_fake_dataset[:, -1] = y
    p_fake_dataset[:, -1] = p
    # inverse transform and then select the right field
    y_unscaled = scaler.inverse_transform(y_fake_dataset)[:, -1]
    p_unscaled = scaler.inverse_transform(p_fake_dataset)[:, -1]

    return y_unscaled, p_unscaled


def prepare_for_train(model_name, model_settings, create_model, parameter_dict):
    db_progress_file = path_join(["Progress_db_train.pkl"])

    if db_progress_file.is_file():
        db_progress_dict = pd.read_pickle(db_progress_file)

        if model_name not in db_progress_dict:
            db_progress_dict[model_name] = []
    else:
        db_progress_dict = {model_name: []}

    runs = model_settings.runs

    # create unique run folder
    train_db_path = create_db_path(model_name, training=True)
    make_dir(train_db_path)

    # Fetch model arguments
    # model_args_dict = {model_name : inspect.getfullargspec(create_model)[0]}
    model_args = inspect.getfullargspec(create_model)[0]
    logger.info('--> The current model has the following hyper parameter: ' + ", ".join(arg for arg in model_args))

    logger.info('--> Fetching randomized hyper parameter')
    par = get_random_parameters(runs, parameter_dict)

    # save parameter lists
    par_list_file = path_join([train_db_path, "random parameter of %s for %s runs.pkl" % (model_name, runs)])
    pickle.dump(par, open(par_list_file, "wb"))

    # save settings
    # current_run_settings = vars(model_settings)
    model_settings.model_name = model_name
    model_settings.train_db_path = str(train_db_path)
    model_settings.parameter_path = str(par_list_file)
    model_settings.model_args = model_args
    model_settings.train_db_prog_file = str(db_progress_file)

    save_to_yaml(model_settings, train_db_path, "settings current run.yml")

    return model_settings, db_progress_dict, par


def multi_run_wrapper(args, x):
    if type(args[-1]) == str:
        if args[-1] == "test":
            test_model(*args[:-1], x)
        elif args[-1] == "train":
            train_model(*args[:-1], x)
    else:
        if len(args) > 4:
            write_db_train(*args, x)
        else:
            write_db_test(*args, x)


def write_db_train(current_run_settings, par, db_progress_dict, train_norm, val_norm, run=None):
    train_db_path = current_run_settings.train_db_path
    model_name = current_run_settings.model_name
    d = current_run_settings.d

    db_name = "db_run_(%s) of (%s).hdf5" % (run + 1, current_run_settings.runs)
    mod_args = {}
    for k in list(par):
        k_ = k[:-5]  # cut off "_list"
        v_ = par[k][run]
        mod_args[k_] = v_

    mod_args["db_name"] = db_name
    save_to_yaml(mod_args, train_db_path, "parameter" + db_name.split(".")[0][2:] + ".yml")
    if db_name in db_progress_dict[model_name]:
        logger.info('--> Training database %s already created ...' % db_name)

    else:
        out_file = path_join([train_db_path, db_name])
        t = mod_args["T"]
        logger.info('--> Creating database %s for training and validation ...' % db_name)
        data_converter({"train_norm": train_norm, "val_norm": val_norm}, t, d, out_file)


def prepare_for_testing(current_run_settings):
    db_progress_file = path_join(["Progress_db_test.pkl"])
    train_db_path = current_run_settings.train_db_path
    model_name = current_run_settings.model_name

    if db_progress_file.is_file():
        db_progress_dict = pd.read_pickle(db_progress_file)
        if model_name not in db_progress_dict:
            db_progress_dict[model_name] = []
    else:
        db_progress_dict = {model_name: []}

    test_db_path = create_db_path(model_name, testing=True)
    make_dir(test_db_path)

    current_run_settings.test_db_path = str(test_db_path)
    current_run_settings.test_db_prog_file = str(db_progress_file)

    save_to_yaml(current_run_settings, train_db_path, "settings current run.yml")

    return db_progress_dict, current_run_settings


def update_db_progress_dict(db_list, db_path):
    # recheck existing db files
    existing_dbs = [os.path.basename(file) for file in glob.glob(str(path_join([db_path, "*.hdf5/"])))]
    for f in existing_dbs:
        if f not in db_list:
            db_list.append(f)

    # remove non-existing dbs
    _ = db_list
    for f in db_list:
        if f not in existing_dbs:
            _.remove(f)

    return sorted_nicely(_)


def process_train_data(model_settings, mod_import, skip_db_train, model_name, train_norm, val_norm):
    model_settings, db_progress_dict, par = prepare_for_train(model_name, model_settings,
                                                              mod_import.create_model,
                                                              mod_import.parameterDict)

    train_db_path = model_settings.train_db_path
    if not skip_db_train:
        logging.info("Creating training databases for %s ..." % model_name)

        # update db_progress_dict
        db_progress_dict[model_name] = update_db_progress_dict(db_progress_dict[model_name], train_db_path)

        args = [model_settings, par, db_progress_dict, train_norm, val_norm]

        with tqdm(total=model_settings.runs, file=sys.stdout) as pbar:

            for run in range(model_settings.runs):
                write_db_train(*args, run=run)
                pbar.update(1)

        # update db_progress_dict
        db_progress_dict[model_name] = update_db_progress_dict(db_progress_dict[model_name], train_db_path)

        pickle.dump(db_progress_dict, open(model_settings.train_db_prog_file, "wb"))
        success()
    else:
        logger.info('--> Training database creation is in debug mode and will be skipped!')

    return model_settings


def process_testing_data(model_settings, skip_db_test, model_name, test_norm, train_db_path):
    db_progress_dict, model_settings = prepare_for_testing(model_settings)
    if not skip_db_test:
        logger.info("Creating testing databases for model %s ..." % model_name)

        # update db_progress_dict
        db_progress_dict[model_name] = update_db_progress_dict(db_progress_dict[model_name],
                                                               model_settings.test_db_path)

        # get all parameter file for imported model for this specific run
        par_path_list = [f for f in glob.glob(str(path_join([train_db_path, "parameter*.yml/"])))]
        par_path_list = sorted_nicely([f for f in par_path_list
                                       if int(model_settings.runs) == int(re.findall('\(+(.*?)\)', f)[-1])])

        args = [par_path_list, model_settings, test_norm, db_progress_dict]

        with tqdm(total=model_settings.runs, file=sys.stdout) as pbar:
            for run in range(model_settings.runs):
                write_db_test(*args, run=run)
                pbar.update(1)

        # update db_progress_dict
        db_progress_dict[model_name] = update_db_progress_dict(db_progress_dict[model_name],
                                                               model_settings.test_db_path)

        pickle.dump(db_progress_dict, open(model_settings.test_db_prog_file, "wb"))
        success()
    else:
        logger.info('--> Test database creation is in debug mode and will be skipped!')

    return model_settings


def write_db_test(par_path_list, cur_run_settings, test_norm, db_progress_dict, run=None):
    par = load_yaml_file(par_path_list[run], silent=True)
    t = par["T"]
    d = cur_run_settings.d
    db = par["db_name"]

    if db in db_progress_dict[cur_run_settings.model_name]:
        logger.info('--> Testing database %s already created ...' % db)

    else:
        logger.info('--> Generating %s for testing ...' % db)
        out_file = path_join([cur_run_settings.test_db_path, db])
        data_converter({"test_norm": test_norm}, t, d, out_file)


def load_train_progress(train_progress_file):
    """ loads train progress file"""
    train_progress_dict = pd.read_pickle(train_progress_file)
    return train_progress_dict


def set_seed(seed):
    """sets seed for all used randomisation functions"""
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    random.seed(seed)


def get_random_parameters(runs, parameter_dict):
    """Runs sampler from spotpy and returns a dict with ramdom parameter lists for all dynamic hyperparameter"""

    parameter = run_spotpy(parameter_dict, runs)  # parameter is used within exec function
    par = {}
    for k, v in parameter_dict.items():
        if v[-1] == int:
            par["%s_list" % k] = [int(x) for x in np.around(parameter['par%s_list' % k])]
        elif v[-1] == float:
            par["%s_list" % k] = [float(x) for x in np.around(parameter['par%s_list' % k], 2)]

    return par


def init_logging():
    console_logging = load_yaml_file("globalSettings.yml", silent=True)["console_logging"]

    # initiate logging
    logging.basicConfig()

    log_file = path_join(["example.log"])

    logger_ = logging.getLogger('mylog')
    logger_.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file, 'a', 'utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', "%d/%m/%Y %H:%M:%S"))
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    if console_logging:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.WARNING)
    logger_.addHandler(file_handler)
    logger_.addHandler(console_handler)

    logger_.propagate = False

    return logger_


def success():
    logger.info('--> Done!')


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def train_model(mod_import, model_settings, progress_df, db, model_name, bs, epochs):
    """

    :return:
    """

    db_name = Path(db).stem
    is_trained = check_progress_df(progress_df, model_name, db_name, bs, epochs)

    if is_trained:
        logger.info('''--> DB: %s with BS: %s and EPOCHS: %s is already trained and
                        will be skipped''' % (db_name, bs, epochs))
        return


    with h5py.File(db, 'r') as db_:
        x = db_["X_train"][...]
        y = db_["Y_train"][...]
        x_val = db_["X_val"][...]
        y_val = db_["Y_val"][...]

        train_shape = x.shape
        # val_shape = x_val.shape
        x = np.concatenate((x, x_val,), axis=0)
        y = np.concatenate((y, y_val,), axis=0)

        train_indices = np.arange(train_shape[0])
        val_indices = np.arange(len(train_indices), x.shape[0])

        del x_val
        del y_val
        clean_up()


    class DataGen(tf.keras.utils.Sequence):
        def __init__(self, index_map, batch_size):
            self.x = x
            self.y = y
            self.index_map = index_map
            self.batch_size = batch_size

        def __getitem__(self, index):
            x_batch = self.x[self.index_map[
                             index * self.batch_size: (index + 1) * self.batch_size
                             ]]

            y_batch = self.y[self.index_map[
                             index * self.batch_size: (index + 1) * self.batch_size
                             ]]
            return x_batch, y_batch

        def __len__(self):
            return len(self.index_map) // self.batch_size

        def on_epoch_end(self):
            'Updates indexes after each epoch'
            np.random.shuffle(self.index_map)

        # def on_epoch_end(self):
        #     np.random.shuffle(self.indices)

    # def lr_scheduler(epoch, warmup_epochs=5, decay_epochs=15, initial_lr=1e-6, base_lr=1e-3, min_lr=5e-5):
    def lr_scheduler(epoch, lr, warmup_epochs=2, decay_epochs=11, initial_lr=1e-6, base_lr=5e-4, min_lr=5e-5):

        if epoch <= warmup_epochs:
            pct = epoch / warmup_epochs

            return ((base_lr - initial_lr) * pct) + initial_lr

        if warmup_epochs < epoch < warmup_epochs + decay_epochs:
            pct = 1 - ((epoch - warmup_epochs) / decay_epochs)

            return ((base_lr - min_lr) * pct) + min_lr

        return min_lr

    clean_up()
    # db = db_list[run]

    # Training
    logger.info('Staring model training with: ...')
    logger.info('--> DB: %s' % db_name)
    logger.info('--> BS: %s' % bs)
    logger.info('--> Epochs: %s' % epochs)

    model_par_dict = {}

    # create output folder
    train_path = create_train_path(model_settings.model_name, db_name, epochs, bs)
    make_dir(train_path)

    # load parameter for this specific run
    mod_args = load_yaml_file(path_join([model_settings.train_db_path, "parameter%s.yml" % db_name[2:]]))
    mod_args["lr"] = model_settings.lr
    mod_args["D"] = model_settings.d
    logger.info(f"--> {mod_args}")

    model_par = [mod_args[i] for i in inspect.getfullargspec(mod_import.create_model)[0]]
    model = mod_import.create_model(*model_par)

    start_time = time.time()

    adapt_learning_rate = LearningRateScheduler(lr_scheduler, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)

    callbacks_list = [adapt_learning_rate, early_stop]  # , PlotLearning()]

    go_ahead = True
    try:
        # with h5py.File(db, 'r') as db_:
        #     logger.info(f"DB shape: {db_['X_train'][...].shape}")

        train_gen = DataGen(train_indices, bs)
        val_gen = DataGen(val_indices, bs)

        r = model.fit(
            # db_["X_train"][...],
            # db_["Y_train"][...],
            train_gen,
            epochs=epochs,
            steps_per_epoch=train_shape[0] // bs,
            # batch_size=bs,
            verbose=1,
            shuffle=True,
            validation_data=val_gen,
            # validation_data=(db_["X_val"][...], db_["Y_val"][...]),
            callbacks=callbacks_list
        )

        save_to_pickle(train_path, r.history, "trainHistoryDict.pkl")
        save_model(model, model_settings, train_path)

        run_time = timer(start_time, time.time())
        logger.info(f"Run time: {run_time}")
        logger.info('--> Model training was successfully, processing time: %s' % run_time)

    except Exception as e:
        tmp_file = "failed_dbs.pkl"
        if os.path.isfile("failed_dbs.pkl"):
            with open("failed_dbs.pkl", "rb") as f:
                tmp_db = pickle.load(f)
                if db_name not in tmp_db:
                    tmp_db.append(db_name)
        else:
            tmp_db = [db_name]

        with open(tmp_file, 'wb') as outfile:
            pickle.dump(tmp_db, outfile)

        logger.error(f"{e}")
        go_ahead = False
        run_time = None

    del x
    del y
    clean_up()

    if go_ahead:
        used_settings = vars(model_settings).copy()
        used_settings["bs"] = bs
        used_settings["epochs"] = epochs
        used_settings["db_name"] = db_name + ".hdf5"
        used_settings["run_time"] = run_time

        for k, v in model_par_dict.items():
            used_settings[k] = v

        tmp_gs = used_settings["model_args"]
        new_ma = {}
        for ma in tmp_gs:
            new_ma[ma] = mod_args[ma]
        new_ma["t"] = mod_args["T"]
        used_settings["model_args"] = new_ma

        save_model_parameter(train_path, used_settings)

        f_name = model_name + "-" + db_name + "-" + str(bs) + "-" + str(epochs)
        pickle.dump([model_name, db_name, bs, epochs, np.nan],
                    open(path_join(["data", "tmp", f_name + ".pkl"]), "wb"))

    run_progress = db_name.split("_")[-1].split()[0][1:-1]


def save_model(model, glob_settings, train_path):
    # save model weights and parameter
    model.save(path_join([train_path, glob_settings.model_name]))
    logger.info('--> Trained model was saved successfully')


def save_to_pickle(train_path, data, outname):
    # saves dicts to pickle
    with open(path_join([train_path, outname]), 'wb') as outfile:
        pickle.dump(data, outfile)
    if "history" in outname:
        logger.info('--> Training history was saved successfully')


def save_model_parameter(train_path, glob_settings_dict):
    # save all used parameter as yaml
    save_to_yaml(glob_settings_dict, train_path, "parameter.yml")
    logger.info('--> Used parameter were saved successfully')


def test_model(db, model_settings, progress_df, epochs, bs):
    m_name = model_settings.model_name
    db_name = Path(db).stem
    db = path_join([model_settings.test_db_path, db_name + ".hdf5"])

    train_path = create_train_path(m_name, db_name, epochs, bs)
    if os.path.isdir(path_join([train_path, m_name])):

        is_tested = check_progress_df(progress_df, m_name, db_name, bs, epochs, testing=True)

        if not is_tested:

            model = load_model(path_join([train_path, m_name]), compile=False)
            model.compile(loss=CustomLoss(), optimizer=tf.keras.optimizers.Adam(learning_rate=model_settings.lr))
            # todo try Nadam

            with h5py.File(db, 'r') as data:
                x = data["X_test"][...]
                y = data["Y_test"][...].flatten()

            prediction = model.predict(x, verbose=0)[:, 0].flatten()  # flatten was needed now

            n = model_settings.test_samples
            split = int(len(y) / n)

            # unscale data
            y_unscaled, p_unscaled = unscale(y, prediction, n)

            output_path = path_join([train_path, "testing"])

            make_dir(output_path)

            pickle.dump(y_unscaled, open(path_join([output_path, "test_observation_unscaled.pkl"]), "wb"))
            pickle.dump(p_unscaled, open(path_join([output_path, "test_prediction_unscaled.pkl"]), "wb"))

            metrics_list = []
            kge_list = []
            for i in range(n):

                y_sample = y_unscaled[i * split:(i + 1) * split]
                p_sample = p_unscaled[i * split:(i + 1) * split]

                if len(y_sample) * n != len(y):
                    logger.error(
                        "--> Test samples have no equal sizes, check 'test_samples' within global Settings")
                    break

                metrics = calc_evaluation_metrics(y_sample, p_sample)
                kge = metrics["kge"]
                kge_list.append(kge)
                metrics_list.append(metrics)

                # line plot
                if model_settings.line_plotting:
                    target_df = pd.DataFrame(list(zip(y_sample, p_sample)), columns=['observed', 'predicted'])
                    logger.info('--> Plotting line charts ...')
                    river_ids = get_river_ids(model_settings.path_to_testing_df)
                    line_plotting(target_df, i + 1, output_path, kge, river_ids)

                # save results
                pickle.dump(metrics_list, open(path_join([output_path, "evaluation_metrics_test-results.pkl"]), "wb"))

                if i == n - 1:
                    logger.info('--> Model testing was successful')
                    kge_df = pd.DataFrame(kge_list, columns=['KGE'])
                    # plot results
                    if not kge_df.isnull().values.any():
                        violin_plotting(output_path, kge_df)
                    else:
                        logger.error('--> KGE could not be calculated for all testing sites!')

                clean_up()

            f_name = m_name + "-" + db_name + "-" + str(bs) + "-" + str(epochs)
            is_tested = check_progress_df(progress_df, m_name, db_name, bs, epochs, testing=True)
            out = [m_name, db_name, bs, epochs, True]

            pickle.dump(out, open(path_join(["data", "tmp", f_name + ".pkl"]), "wb"))

        else:
            logger.info('--> Testing was already processed')
    else:
        logger.info(f'--> No trained Model found in "{path_join([train_path, m_name])}"')


def get_river_ids(file, with_index=False):
    with open(file, "rb") as f:
        df = pickle.load(f)

    # print(df.to_string())

    river_ids = []
    group = None
    for name, group in df.groupby("gauge_id"):
        river_ids.append(name)

    if not with_index:
        return river_ids

    else:
        return river_ids, group.index


def line_plotting(df, i, path, kge, river_ids):
    out_dir = path_join([path, "LinePlots"])
    make_dir(out_dir)

    fig = plt.figure(figsize=(16, 8))
    ax = sns.lineplot(data=df)

    max_y = df.max().max()
    ax.set_title("Comparing observed and predicted Discharge for Gauge %s" % river_ids[i - 1])
    ax.set_ylabel("Discharge in mm")
    ax.set_xlabel("Days")
    ax.text(0, max_y * 0.9, 'KGE: ' + str(np.round(kge, 2)), fontsize=10)  # add text

    # fig = ax.get_figure()
    fig.savefig(path_join([out_dir, 'Discharge vs. Prediction River {}.png'.format(river_ids[i - 1])]), dpi=240)
    plt.close(fig)


def violin_plotting(path, data):
    sns.set_theme(style="whitegrid")
    sns.set_context("paper", rc={"lines.linewidth": 2,
                                 'xtick.labelsize': 18.0,
                                 'ytick.labelsize': 18.0,
                                 'legend.fontsize': 18.0,
                                 'axes.labelsize': 19.0,
                                 'axes.titlesize': 19.0,
                                 })
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.violinplot(data=data, x="KGE", cut=0, split=False, inner='quart', orient="h", ax=ax)  # bw=.2)
    ax1 = sns.swarmplot(data=data, x="KGE", color="brown", edgecolor="black", size=7, linewidth=1)

    ax.set_title("KGE Distributuion of best run, Mean KGE: " + str(np.round(data.KGE.mean(), 2)), pad=40,
                 fontdict={'fontsize': 24})
    # ax.set_ylabel("Discharge in mm", labelpad=20)
    ax.set_xlabel("KGE", labelpad=20)
    ax.set_ylabel(None)
    ax.set(xlim=(0, 1))
    plt.xticks(np.arange(0, 1.1, 0.1))

    fig.tight_layout()
    fig.subplots_adjust(top=0.87)
    
    fig.savefig(path_join([path, "KGE.png"]), dpi=240)
    plt.close(fig)


def check_progress_df(df, model_name, db, bs, epochs, testing=False):
    a = df[df['modelName'] == model_name]
    b = a[a['dbName'] == db]
    c = b[b['batchSize'] == bs]
    d = c[c['epochs'] == epochs]
    f = d[d['testing'] == True]

    if any((len(a.index) == 0, len(b.index) == 0, len(c.index) == 0, len(d.index) == 0)):
        return False
    else:
        if testing:
            if len(f.index) == 0:
                return False
            else:
                return True
        else:  # training
            return True


def create_train_path(model_name, db_name, epochs, bs):
    train_path = path_join(["data", "training", model_name, db_name,
                            "epoch = " + str(epochs), "batchsize = " + str(bs)])
    return train_path


def create_db_path(model_name, training=False, testing=False):
    if training:
        db_path = path_join(["data", "db for training and validation", model_name])
    elif testing:
        db_path = path_join(["data", "db for testing", model_name])
    else:
        db_path = None

    return db_path


def sorted_nicely(l):
    """ Sorts the given iterable in the way that is expected.

    Required arguments:
    l -- The iterable to be sorted.

    """

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def clean_up():
    gc.collect()
    K.clear_session()
    tf.compat.v1.reset_default_graph()


def update_progress(dir, df, out_file):
    col_names = list(df)
    files = glob.glob(str(path_join([dir, "*.pkl/"])))
    if len(files) > 0:
        for file in [f for f in files]:
            _ = pickle.load(open(file, "rb"))
            df_tmp = df[((df[col_names[0]] == _[0]) & (df[col_names[1]] == _[1]) & (df[col_names[2]] == _[2]) & (
                    df[col_names[3]] == _[3]))]
            if len(df_tmp) == 0:
                df.loc[len(df)] = _
            if _[4] == True and len(df_tmp) > 0:
                idx = df_tmp.index[0]
                df.loc[idx, "testing"] = True

            try:
                os.remove(file)
            except Exception as e:
                logger.error("Could not delete temp file: %s" % file)
                logger.error(e)

        pickle.dump(df, open(out_file, "wb"))

    return df


def calc_evaluation_metrics(evaluation, simulation):
    kge = spotpy.objectivefunctions.kge(evaluation, simulation)
    nse = spotpy.objectivefunctions.nashsutcliffe(evaluation, simulation)
    pbias = spotpy.objectivefunctions.pbias(evaluation, simulation)
    mae = spotpy.objectivefunctions.mae(evaluation, simulation)
    r2 = spotpy.objectivefunctions.rsquared(evaluation, simulation)

    return {"kge": kge, "nse": nse, "pbias": pbias, "mae": mae, "r2": r2}


logger = logging.getLogger("root")

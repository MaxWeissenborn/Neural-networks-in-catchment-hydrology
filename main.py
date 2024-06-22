from scripts.custom_functions import log

logger = log.init_logging("root")
from scripts.custom_functions.general import *
from types import SimpleNamespace
import importlib
import copy
import pandas as pd
import itertools
from tqdm import tqdm
import shutil


desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 20)


def main():
    # split train df in train and validation sets
    logger.info('Splitting dataframe in train and val sets ...')
    train, validate = split_to_train_and_validation(df)
    success()

    # number of features
    d = train.shape[1] - 2  # - gauge_id + target todo: this is hardcoded
    globSettings.d = d

    # normalize train data
    if globSettings.testing:
        logger.info('Normalizing train, validation and test data sets ...')
    else:
        logger.info('Normalizing train and validation data sets ...')
    train_norm, val_norm, test_norm = normalize(train, validate, df_test, globSettings, model_suffix)
    success()

    # handle train progress from previous runs
    logger.info('Loading train progress file')
    train_prog_file = path_join(["Progress_training.pkl"])
    if train_prog_file.is_file():
        try:
            progress_df = load_train_progress(train_prog_file)
            logger.info('--> Train progress file was loaded successfully')
        except:
            logger.error("--> Train progress file exists, but couldn't be loaded")
            sys.exit()
    else:
        logger.info('--> No training progress from previous runs was found, starting all over')
        progress_df = pd.DataFrame(columns=['modelName', 'dbName', 'batchSize', 'epochs', 'testing'])

    # searching for model files
    logger.info('Searching for models ...')
    model_names = [Path(models).stem for models in glob.glob(str(path_join([Path.cwd(),
                                                                            "scripts", "models", "*.py/"])))]
    # create a temp dir
    tmp_dir = path_join(["data", "tmp"])
    make_dir(tmp_dir)

    # start process
    if model_names:
        bs_list = globSettings.bs
        epochs_list = globSettings.epochs

        for modelName in model_names:

            model_settings = copy.copy(globSettings)
            if modelName not in model_settings.run_models and len(model_settings.run_models) > 0:
                continue
            try:
                mod_import = importlib.import_module("scripts.models.%s" % modelName)
                logger.info('--> Model "%s" was imported successfully' % modelName)

            except ImportError as e:
                logger.error('--> Failed to import %s' % modelName)
                logger.error(e)
                continue

            modelName = f"{modelName} {model_suffix}"

            # loading model information into working namespace
            model_settings.description = mod_import.description

            # process train/val db
            model_settings = process_train_data(model_settings, mod_import, skip_db_train, modelName, train_norm,
                                                val_norm)
            train_db_path = model_settings.train_db_path

            # process test db
            model_settings = process_testing_data(model_settings, skip_db_test, modelName,
                                                  test_norm, train_db_path)

            # get all databases for imported model for this specific run
            db_list = [db for db in glob.glob(str(path_join([train_db_path, "*.hdf5/"])))]
            db_list = sorted_nicely([f for f in db_list
                                     if int(model_settings.runs) == int(re.findall('\(+(.*?)\)', f)[-1])])

            logger.info("Starting training for model: %s" % modelName)

            progress_df = update_progress(tmp_dir, progress_df, train_prog_file)
            model_settings.path_to_training_df = train_df_path
            model_settings.path_to_testing_df = test_df_path

            if not skip_training:

                with tqdm(total=model_settings.runs * len(bs_list) * len(epochs_list), file=sys.stdout) as pbar:
                    for db, bs, epochs in itertools.product(db_list, bs_list, epochs_list):
                        train_args = [mod_import, model_settings, progress_df, db, modelName, bs, epochs]
                        train_model(*train_args)

                        pbar.update(1)

                success()

                progress_df = update_progress(tmp_dir, progress_df, train_prog_file)

            else:
                logger.info('--> Model training is in debug mode')

            if not skip_test:

                # Testing with untrained data
                if model_settings.testing:

                    logger.info('Processing validation with test set for model: %s' % modelName)

                    with tqdm(total=model_settings.runs * len(bs_list) * len(epochs_list), file=sys.stdout) as pbar:
                        for db, bs, epochs in itertools.product(db_list, bs_list, epochs_list):
                            test_args = [db, model_settings, progress_df, epochs, bs]
                            test_model(*test_args)

                            pbar.update(1)

                        progress_df = update_progress(tmp_dir, progress_df, train_prog_file)

                else:
                    logger.info('Testing was globally deactivated, therefore it will be skipped')
            else:
                logger.info('--> Model validation with test set is in debug mode and will be skipped')

            if model_settings.delete_data_bases:
                try:
                    logger.info(f"Deleting training database for model: {modelName}")
                    shutil.rmtree(model_settings.train_db_path, ignore_errors=False, onerror=None)
                except Exception as e:
                    logger.error(f"Can't delete training data bases. Reason: {e}")
                try:
                    logger.info(f"Deleting testing database for model: {modelName}")
                    shutil.rmtree(model_settings.test_db_path, ignore_errors=False, onerror=None)
                except Exception as e:
                    logger.error(f"Can't delete testing data bases. Reason: {e}")

    else:
        logger.error("No models found!")
        sys.exit()

    logger.info('All done, program is shutting down')


if __name__ == '__main__':

    # debug variables
    skip_db_train = True  # includes validation db
    skip_db_test = True
    skip_training = False
    skip_test = False
    version = 1.1

    # get logger
    logger = logging.getLogger('mylog')

    # load global settings
    globSettings = SimpleNamespace(**load_yaml_file())

    if globSettings.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU

    # init seed
    set_seed(globSettings.seed)

    for model_suffix, train_df_path in globSettings.path_to_training_df.items():
        logger.info('Loading training dataframe ...')
        with open(train_df_path, "rb") as f:
            df = pickle.load(f)
        success()

        # set target column as the last column of data frames
        df['target'] = df[globSettings.target_name]
        df.drop(labels=[globSettings.target_name], axis=1, inplace=True)

        # load test file if globally selected
        if globSettings.testing:
            logger.info('Loading testing dataframe ...')
            test_df_path = globSettings.path_to_testing_df[model_suffix]
            with open(test_df_path, "rb") as f:
                df_test = pickle.load(f)
            success()

            df_test['target'] = df_test[globSettings.target_name]
            df_test.drop(labels=[globSettings.target_name], axis=1, inplace=True)
        else:
            df_test = None

        main()

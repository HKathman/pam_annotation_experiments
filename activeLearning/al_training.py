import os
import random
import pickle
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import config as cfg
import utils
from activeLearning import al_sampling
from classificationHead import linear_classification_head


def train_active_learning_models():
    # create folders
    utils.create_folder(cfg.path_results_dataset_al_experiments_history)
    utils.create_folder(cfg.path_results_dataset_al_experiments_logs)
    utils.create_folder(cfg.path_results_dataset_al_experiments_predictions)

    # create initial training dataset for all random seeds
    al_sampling.initial_training_data_devision()

    # load tag and batch split data
    df_initial_split = pd.read_pickle(cfg.path_results_dataset_al / 'initial_training_set.pickle')

    # load data
    file_embedding = cfg.selected_embedding + '.npy'
    x_data_original = np.load(cfg.path_results_dataset_tl_embedding / file_embedding)

    # get label for data
    y_data_original = utils.get_y_original_from_metadata()

    # initialise df to save annotation time
    df_computation_time = pd.DataFrame()
    df_computation_index = -1

    # iterate over selected_uncertainty_sampling_strategies
    for sampling_strategy in cfg.selected_sampling_strategies:

        # update computation index
        df_computation_index = df_computation_index + 1

        # iterate over random seeds
        for random_seed in cfg.random_seed_value:

            # create file name
            file_name = cfg.selected_dataset + '-' + cfg.selected_embedding + '-' + sampling_strategy + '-random_seed-' + str(random_seed)
            file_df = file_name + '.pickle'
            if os.path.exists(cfg.path_results_dataset_al_experiments / file_df):
                continue

            print('Active Learning Experiment | ', sampling_strategy, ' | random seed: ', random_seed)

            # set random seed
            random.seed(random_seed)
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)

            # initialize output df
            tag_col = utils.get_tag_iteration_column(0)
            df = pd.DataFrame()
            df[tag_col] = df_initial_split[f'tag_seed_{random_seed}']

            # create model
            model = linear_classification_head.create_model(x_data_original, y_data_original)

            # initialize progress bar (for display)
            max_training_samples = min(len(df[df[tag_col].isin((cfg.tag_train, cfg.tag_validation, cfg.tag_unlabelled))]),
                                       cfg.selected_max_training_samples)
            pbar = tqdm(total=max_training_samples)

            # iterate while there is lees than max training data
            iteration = 0
            y_pred = np.empty((0, 0))

            # start timer for saving computational cost
            start_time = time.time()

            while True:

                # defragment dataframe
                df = df.copy()

                # get current tag column
                tag_col = utils.get_tag_iteration_column(iteration)

                # select samples for training
                df = al_sampling.define_next_training_set(df, sampling_strategy, iteration, model, y_pred, y_data_original, x_data_original)

                # update progress bar
                pbar.update(len(df[(df[tag_col] == cfg.tag_train) | (df[tag_col] == cfg.tag_validation)]) - pbar.n)

                # get training and validation data
                array_tag = df[tag_col].to_numpy()
                x_train = x_data_original[array_tag == cfg.tag_train]
                y_train = y_data_original[array_tag == cfg.tag_train]

                x_val = x_data_original[array_tag == cfg.tag_validation]
                y_val = y_data_original[array_tag == cfg.tag_validation]

                # shuffle data
                shuffled_indices = np.random.permutation(len(x_train))
                x_train = x_train[shuffled_indices]
                y_train = y_train[shuffled_indices]

                # convert data to tf inputs
                x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
                y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

                x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
                y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)

                # train model
                current_log_dir = file_name + '-iteration-' + str(iteration)
                log_dir = cfg.path_results_dataset_al_experiments_logs / current_log_dir
                history = linear_classification_head.train_model(model, x_train, y_train, x_val, y_val, log_dir=log_dir)

                # save history
                file_history = current_log_dir + '-history.pkl'
                path_history = cfg.path_results_dataset_al_experiments_history / file_history
                with open(path_history, 'wb') as file:
                    pickle.dump(history.history, file)

                # get and save predictions
                x_all = tf.convert_to_tensor(x_data_original, dtype=tf.float32)
                y_pred = model(x_all)
                y_pred = y_pred.numpy()
                file_prediction = current_log_dir + '-ypred.npy'
                np.save(cfg.path_results_dataset_al_experiments_predictions / file_prediction, y_pred)

                if len(df[df[tag_col].isin((cfg.tag_train, cfg.tag_validation))]) >= max_training_samples:
                    break
                iteration = iteration + 1

            # save processing time
            end_time = time.time()
            elapsed_time = end_time - start_time
            df_computation_time.loc[df_computation_index, 'sampling_strategy'] = sampling_strategy
            df_computation_time.loc[df_computation_index, random_seed] = elapsed_time
            print(df_computation_time)
            df_computation_time.to_csv(cfg.path_results_dataset_al / 'timings.csv')

            # Close the progress bar
            pbar.close()

            # save df
            df.to_pickle(cfg.path_results_dataset_al_experiments / file_df)


if __name__ == '__main__':
    train_active_learning_models()














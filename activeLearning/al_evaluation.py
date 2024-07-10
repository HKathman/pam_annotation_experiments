import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as cfg
import utils


def evaluate_active_learning_models():

    # create folder
    utils.create_folder(cfg.path_results_dataset_al_evaluation)

    # define list of datatags
    data_tags = [cfg.tag_train, cfg.tag_unlabelled, cfg.tag_validation, cfg.tag_test]

    # get list of all experiment files
    all_al_experiments = [experiment for experiment in os.listdir(cfg.path_results_dataset_al_experiments)
                          if experiment.endswith('.pickle')]

    # get y_true_original
    y_true_original = utils.get_y_original_from_metadata()

    # get list of all unique al sampling methods
    al_methods = list(set([experiment.split('-')[-3] for experiment in all_al_experiments]))

    # iterate over all unique sampling methods
    for al_method in al_methods:

        # check if output already exists
        file_name = al_method + '.pickle'
        if os.path.exists(cfg.path_results_dataset_al_evaluation / file_name):
            continue

        # get files that include the al method (from different random seeds)
        experiments = [experiment for experiment in all_al_experiments if al_method in experiment]

        # initialise lists for output
        list_random_seed = []
        list_iteration = []
        list_data_tag = []
        list_nr_samples = []
        list_tp = []
        list_tn = []
        list_fp = []
        list_fn = []

        # print output
        print('Evaluate active learning method: ', al_method)
        pbar = tqdm(total=len(experiments))

        # iterate over all experiments (from random seeds)
        for experiment in experiments:

            # load dataframe with the data tag information
            df_data_split = pd.read_pickle(cfg.path_results_dataset_al_experiments / experiment)

            # define random seed
            base_name = os.path.splitext(experiment)[0]
            random_seed = base_name.split('-')[-1]

            # increase pbar output
            pbar.update(1)

            # iterate over all active learning iterations (different nr of training samples)
            for iteration_column in df_data_split.columns:

                # create file name of prediction values
                iteration_nr = iteration_column.split('_')[-1]
                y_pred_file_name = base_name + '-iteration-' + iteration_nr + '-ypred.npy'

                # load prediction values and set threshold 0.5
                y_pred_original = np.load(cfg.path_results_dataset_al_experiments_predictions / y_pred_file_name)
                y_pred_original[y_pred_original > 0.5] = 1
                y_pred_original[y_pred_original <= 0.5] = 0

                # iterate over data tags
                for data_tag in data_tags:

                    # get indices of data with the current data_tag
                    relevant_indices = df_data_split[df_data_split[iteration_column] == data_tag].index

                    # get relevant y_pred and y_true values
                    y_true = y_true_original[relevant_indices]
                    y_pred = y_pred_original[relevant_indices]

                    # compute properties and append to list
                    list_random_seed.append(random_seed)
                    list_iteration.append(iteration_nr)
                    list_data_tag.append(data_tag)
                    list_nr_samples.append(y_true.shape[0])
                    list_tp.append(np.sum((y_true == 1) & (y_pred == 1), axis=0))
                    list_tn.append(np.sum((y_true == 0) & (y_pred == 0), axis=0))
                    list_fp.append(np.sum((y_true == 0) & (y_pred == 1), axis=0))
                    list_fn.append(np.sum((y_true == 1) & (y_pred == 0), axis=0))

        # close pbar
        pbar.close()

        # create dataframe
        df = pd.DataFrame({'random_seed': list_random_seed,
                           'iteration': list_iteration,
                           'data_tag': list_data_tag,
                           'nr_samples': list_nr_samples,
                           'tp': list_tp,
                           'tn': list_tn,
                           'fp': list_fp,
                           'fn': list_fn})

        # save dataframe
        df.to_pickle(cfg.path_results_dataset_al_evaluation / file_name)

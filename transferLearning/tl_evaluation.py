import os

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

import config as cfg
from transferLearning import tl_training as tlTrain
from classificationHead import linear_classification_head as LinClassificationHead
import utils


def evaluateTransferLearningModels(data_tag, f1_metric):

    utils.createFolder(cfg.path_results_dataset_tl_evaluation)

    df_results_name = 'transfer_learning_' + f1_metric + '_f1_' + data_tag + '.csv'
    if os.path.exists(cfg.path_results_dataset_tl_evaluation / df_results_name):
        return

    # load tag and batch split data
    df_training_split = pd.read_pickle(cfg.path_results_dataset_tl / 'training_split.pickle')

    # get label for data
    y_data_original = utils.getYoriginalFromMetadata()

    # initialize df_f1
    df_f1 = pd.DataFrame({'embedding': pd.Series(cfg.selected_embeddings).explode()})
    df_f1.reset_index(drop=True, inplace=True)
    df_f1['mean'] = None
    df_f1['median'] = None
    df_f1['std'] = None
    for random_seed in cfg.random_seed_value:
        df_f1[random_seed] = None

    # iterate over all models
    for index, row in df_f1.iterrows():
        embedding = row['embedding']

        # load data
        embedding_file = embedding + '.npy'
        x_data_original = np.load(cfg.path_results_dataset_tl_embedding / embedding_file)

        # iterate over random seeds
        f1_all_values_embedding = []
        for random_seed in cfg.random_seed_value:

            print('Compute evaluation: ', data_tag, ' | ', embedding, ' with random seed: ', random_seed)

            # get col tag seed
            col_tag_seed = 'tag_seed_' + str(random_seed)
            array_tag = df_training_split[col_tag_seed].to_numpy()

            # select x and corresponding y data
            x_data = x_data_original[array_tag == data_tag]
            y_data = y_data_original[array_tag == data_tag]
            if y_data.shape[0] == 0:
                continue

            x_input = tf.convert_to_tensor(x_data, dtype=tf.float32)

            # load model
            file_model = embedding + '_seed_' + str(random_seed) + '.keras'
            model = tf.keras.models.load_model(cfg.path_results_dataset_tl_trained_models / file_model,
                                               custom_objects={'ExponentialSoftmaxPooling': LinClassificationHead.ExponentialSoftmaxPooling})

            # inference
            y_pred = model(x_input)

            # convert data
            y_pred = y_pred.numpy()
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0

            # calc F1 macro score
            f1 = f1_score(y_data, y_pred, average=f1_metric, zero_division=np.nan)

            # save f1 score
            df_f1.at[index, random_seed] = f1

            # save f1 score to array
            f1_all_values_embedding.append(f1)

        # calculate mean and std over all random seeds
        df_f1.at[index, 'mean'] = np.round(np.nanmean(f1_all_values_embedding), 3)
        df_f1.at[index, 'median'] = np.round(np.nanmedian(f1_all_values_embedding), 3)
        df_f1.at[index, 'std'] = np.round(np.nanstd(f1_all_values_embedding), 3)

    # save result
    print(df_f1)
    df_f1.to_csv(cfg.path_results_dataset_tl_evaluation / df_results_name)



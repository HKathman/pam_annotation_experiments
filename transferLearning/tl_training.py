import os
import pickle
import random

import pandas as pd
import numpy as np
import tensorflow as tf

import config as cfg
import utils
from classificationHead import linear_classification_head


def train_and_save_transfer_learning_models():

    utils.create_folder(cfg.path_results_dataset_tl_trained_models)
    utils.create_folder(cfg.path_results_dataset_tl_trained_models_logs)

    # get all numpy files from embedding folder
    embeddings = [file for file in os.listdir(cfg.path_results_dataset_tl_embedding) if file.endswith('.npy')]

    # load tag and batch split data
    df_training_split = pd.read_pickle(cfg.path_results_dataset_tl / 'training_split.pickle')

    # get label for data
    y_data = utils.get_y_original_from_metadata()

    # iterate over all embeddings
    for embedding in embeddings:
        # load embedding
        x_data = np.load(cfg.path_results_dataset_tl_embedding / embedding)

        # iterate over all random seeds
        for random_seed in cfg.random_seed_value:

            # freeze random values
            random.seed(random_seed)
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)

            # create output file name
            file_model = os.path.splitext(embedding)[0] + '_seed_' + str(random_seed) + '.keras'
            if os.path.exists(cfg.path_results_dataset_tl_trained_models / file_model):
                continue

            print('CREATE MODEL: ', file_model)

            # get col tag seed
            col_tag_seed = 'tag_seed_' + str(random_seed)

            # get list tag_seed
            array_tag = df_training_split[col_tag_seed].to_numpy()

            # get training and validation data
            x_train = x_data[array_tag == cfg.tag_train]
            y_train = y_data[array_tag == cfg.tag_train]

            x_val = x_data[array_tag == cfg.tag_validation]
            y_val = y_data[array_tag == cfg.tag_validation]

            # convert training and validation data to tensor
            x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
            y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

            x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
            y_val = tf.convert_to_tensor(y_val, dtype=tf.int32)

            # create model
            model = linear_classification_head.create_model(x_data, y_data)

            # train model
            log_dir = cfg.path_results_dataset_tl_trained_models_logs / file_model.split('.')[0]
            history = linear_classification_head.train_model(model, x_train, y_train, x_val, y_val, log_dir=log_dir)

            # save model
            file_history = file_model.split('.')[0] + '_history.pkl'
            path_history = cfg.path_results_dataset_tl_trained_models / file_history
            with open(path_history, 'wb') as file:
                pickle.dump(history.history, file)

            model.save(cfg.path_results_dataset_tl_trained_models / file_model)




if __name__ == '__main__':
    train_and_save_transfer_learning_models()






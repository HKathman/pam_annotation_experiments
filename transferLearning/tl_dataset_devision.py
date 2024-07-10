import os.path
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import config as cfg
import utils


def create_data_tag_for_transfer_learning():

    if os.path.exists(cfg.path_results_dataset_tl / 'training_split.pickle'):
        return

    # load metadata
    file_name = cfg.selected_dataset + '_metadata.pickle'
    df = pd.read_pickle(cfg.path_results_dataset_metadata / file_name)

    # get name of label column
    label_col = [col for col in df.columns if '[' in col and ']' in col][0]

    # get tag column
    tag_col = utils.getTagIterationColumn(0)

    # change unlabelled to training
    df[tag_col] = df[tag_col].replace(cfg.tag_unlabelled, cfg.tag_train)

    # initialize output dataframe
    df_split = pd.DataFrame()

    # fraction of training data for validation split
    val_split = 0.2

    # iterate over random seeds for different training/validation splits (evaluation set stays the same)
    for random_seed in cfg.random_seed_value:
        random.seed(random_seed)
        np.random.seed(random_seed)

        # create label column name
        name_col = 'tag_seed_' + str(random_seed)

        # create new col in df_split
        df_split[name_col] = df[tag_col]

        # stratified validation split if possible, otherwise random split
        label_counts = df[df[tag_col] == cfg.tag_train][label_col].value_counts()

        if label_counts.min() == 1:
            val_df = df[df[tag_col] == cfg.tag_train].sample(frac=val_split)

        else:
            _, val_df = train_test_split(df[df[tag_col] == cfg.tag_train],
                                         test_size=val_split,
                                         stratify=df[df[tag_col] == cfg.tag_train][label_col],
                                         random_state=random_seed)

        # change df_split values at derived indices from train to validation
        df_split.loc[val_df.index, name_col] = cfg.tag_validation

    # save df_split
    df_split.to_pickle(cfg.path_results_dataset_tl / 'training_split.pickle')


if __name__ == '__main__':
    create_data_tag_for_transfer_learning()
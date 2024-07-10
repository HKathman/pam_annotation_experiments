import random
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import config as cfg
import utils


def define_next_training_set(df, sampling_strategy, iteration, original_model, y_pred, y_true, x_data_original):
    # initial training set already exists
    if iteration == 0:
        df = get_validation_data(df, iteration, y_true)
        return df

    # get last and next tag col
    last_tag_col = utils.get_tag_iteration_column(iteration - 1)
    tag_col = utils.get_tag_iteration_column(iteration)

    # prepare next tag_col
    df[tag_col] = df[last_tag_col]
    df[tag_col] = df[tag_col].replace(cfg.tag_validation, cfg.tag_train)

    # get nr of samples to annotate
    nr_samples = cfg.selected_add_training_samples

    # extract sampling variables
    [samp_method, samp_combine, samp_percentage] = sampling_strategy.split('_')
    nr_samp_method = int(int(samp_percentage) / 100 * nr_samples)
    nr_samp_random = nr_samples - nr_samp_method

    # clone model
    model = tf.keras.models.clone_model(original_model)

    # choose sampling method
    if samp_method == 'random' or nr_samp_method >= df[tag_col].value_counts()[cfg.tag_unlabelled]:
        df = sampling_random(df, nr_samp_method, tag_col)
    elif samp_method == 'leastconfidence':
        y_least_confidence = least_confidence(y_pred)
        df = sampling_uncertainty(df, y_least_confidence, nr_samp_method, samp_combine, tag_col)
    elif samp_method == 'ratio':
        y_ratio = ratio(y_pred)
        df = sampling_uncertainty(df, y_ratio, nr_samp_method, samp_combine, tag_col)
    elif samp_method == 'entropy':
        y_entropy = entropy(y_pred)
        df = sampling_uncertainty(df, y_entropy, nr_samp_method, samp_combine, tag_col)
    elif samp_method == 'clustering':
        df = sampling_clustering(df, nr_samp_method, tag_col, x_data_original)
    elif samp_method == 'transferUncertainty':
        df = sampling_transfer_uncertainty(df, iteration, model, nr_samp_method, y_pred, y_true, x_data_original, adapt_rounds=5)
    elif samp_method == 'transferDiversity':
        df = sampling_transfer_diversity(df, iteration, model, nr_samp_method, x_data_original, adapt_rounds=5)
    elif samp_method == 'combineUncertaintyDiversity':
        # combine with filtering (apply ratio_max, select 50% of unlabelled data, apply clustering)
        y_ratio = ratio(y_pred)
        df = sampling_cluster_preselected_samples(df, nr_samp_method, tag_col, x_data_original, y_ratio)
    elif samp_method == 'combineTransferUncertaintyDiversity':
        # combine transfer_uncertainty with clustering, get 50 % from each method
        nr_samples_per_method = int(nr_samp_method / 2)
        df = sampling_transfer_uncertainty(df, iteration, model, nr_samples_per_method, y_pred, y_true, x_data_original, adapt_rounds=5)
        df = sampling_clustering(df, nr_samples_per_method, tag_col, x_data_original)
    elif samp_method == 'combineUncertaintyTransferDiversity':
        # combine transfer_diversity and ratio max, get 50 % from each method
        nr_samples_per_method = int(nr_samp_method / 2)
        y_ratio = ratio(y_pred)
        df = sampling_uncertainty(df, y_ratio, nr_samples_per_method, 'max', tag_col)
        df = sampling_transfer_diversity(df, iteration, model, nr_samples_per_method, x_data_original, adapt_rounds=5)
    elif samp_method == 'combineTransferUncertaintyTransferDiversity':
        # combine transfer_diversity and transfer uncertainty, get 50 % from each method
        nr_samples_per_method = int(nr_samp_method / 2)
        df = sampling_transfer_uncertainty(df, iteration, model, nr_samples_per_method, y_pred, y_true, x_data_original, adapt_rounds=5)
        df = sampling_transfer_diversity(df, iteration, model, nr_samples_per_method, x_data_original, adapt_rounds=5)
    else:
        utils.errorSamplingStrategy(sampling_strategy)

    # add percentage random sampling
    if nr_samp_random != 0:
        df = sampling_random(df, nr_samp_random, tag_col)

    # create validation set
    df = get_validation_data(df, iteration, y_true)

    return df


'''
RANDOM SAMPLING
'''
def sampling_random(df, nr_samples, tag_col):
    # filter unlabelled rows
    df_unlabelled = df[df[tag_col] == cfg.tag_unlabelled]

    # get random indices
    indices = random.sample(list(df_unlabelled.index), min(len(df_unlabelled), nr_samples))

    # change unlabelled elements in df
    df.loc[indices, tag_col] = cfg.tag_train
    return df


'''
UNCERTAINTY SAMPLING
'''
def least_confidence(y_pred):
    # calculates 1- abs(2y - 1) for every node in every sample
    return 1 - np.abs(2 * y_pred - 1)


def ratio(y_pred):
    # calculate 1 / (0.5 + abs(y - 0.5)) - 1
    return 1 / (0.5 + np.abs(y_pred - 0.5)) - 1


def entropy(y_pred):
    # calculate - ( y log_2(y) + (1-y) log_2(1-y)
    epsilon = 1e-10
    return -(y_pred * np.log2(y_pred + epsilon) + (1-y_pred) * np.log2(1-y_pred + epsilon))


def sampling_uncertainty(df, y_uncertainty_array, nr_samp_method, samp_combine, tag_col):

    # get average or max score
    if samp_combine == 'max':
        y_uncertainty_score = np.max(y_uncertainty_array, axis=1)
    else: # samp_combine == 'average'
        y_uncertainty_score = np.mean(y_uncertainty_array, axis=1)

    # get indices with highest uncertainty score and label them (assign train tag)
    indices_not_unlabelled = df[df[tag_col] != cfg.tag_unlabelled].index
    y_uncertainty_score[indices_not_unlabelled] = -1  # (computed score always in range [0,1])
    high_score_indices = np.argsort(-y_uncertainty_score)[:nr_samp_method]

    # if (for a bug reason) some other data than unlabelled data was selected, select all unlabelled data for training
    if (df.loc[high_score_indices, tag_col] != cfg.tag_unlabelled).any():
        df[df[tag_col] == cfg.tag_unlabelled] = cfg.tag_train
    else:
        df.loc[high_score_indices, tag_col] = cfg.tag_train

    return df


'''
Diversity Sampling
'''
def sampling_clustering(df, nr_samp_method, tag_col, x_data):
    # isolate unlabelled data
    df_unlabelled = df[df[tag_col] == cfg.tag_unlabelled]
    indices_unlabelled = df_unlabelled.index

    x_unlabelled = x_data[indices_unlabelled]

    # sample per cluster: 1 outlier, 1 centroid, 3 random (5 per cluster). Calculate nr. of clusters
    nr_outlier = 1
    nr_centroid = 1
    nr_random = 3
    n_cluster = int(nr_samp_method / (nr_outlier + nr_centroid + nr_random))

    # Initialize the KMeans model with the desired number of clusters and the custom distance metric
    kmeans = KMeans(n_clusters=n_cluster, n_init=1)
    kmeans.fit(x_unlabelled)

    # Calculate the distance between each sample and the cluster centers
    distances = kmeans.transform(x_unlabelled)

    # centroids: Find the index of the sample nearest to each cluster center
    centroid_indices = distances.argmin(axis=0)

    # outlier: Find the index of the samples that are farthest away from the nearest cluster centroid (outliers)
    min_distances = np.min(distances, axis=1)
    outlier_indices = np.argsort(min_distances)[-n_cluster:]

    # random samples:
    cluster_assignments = kmeans.predict(x_unlabelled)
    # Find the indices of data points for each cluster
    cluster_indices = [np.where(cluster_assignments == cluster_idx)[0] for cluster_idx in range(kmeans.n_clusters)]
    # Exclude outlier and centroid indices from the cluster_indices
    cluster_indices = [np.setdiff1d(indices, np.concatenate([outlier_indices, centroid_indices])) for indices in
                       cluster_indices]

    random_indices_per_cluster = []
    for indices in cluster_indices:
        random_indices = np.random.choice(indices, size=min(nr_random, len(indices)), replace=False)
        random_indices_per_cluster.append(random_indices)
    # Concatenate the sampled indices from all clusters
    random_indices = np.concatenate(random_indices_per_cluster)

    # annotate selected samples
    df_row_indices = df_unlabelled.index[np.concatenate((centroid_indices, outlier_indices, random_indices))]
    df.loc[df_row_indices, tag_col] = cfg.tag_train

    return df


'''
Transfer active learning
'''
def sampling_transfer_uncertainty(df, iteration, model, nr_samp_method, y_pred, y_true, x_data, adapt_rounds=1):
    # get tag_col and last tag_col
    tag_col = utils.get_tag_iteration_column(iteration)
    last_tag_col = utils.get_tag_iteration_column(iteration - 1)

    # Model preparation: freeze all layers except the last one
    for layer in model.layers[:-1]:
        layer.trainable = False
    # exchange the last layer to have only one node
    model.pop()
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # compile model
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    # binarase y_pred
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    # get new label correct, incorrect
    prediction_result = np.all(y_true == y_pred, axis=1)
    prediction_result = prediction_result.astype(int)

    # Initialise df_validation: Use validation data to train model. Get validation array
    df_validation = df[df[last_tag_col] == cfg.tag_validation].copy()
    df_unlabelled = df[df[last_tag_col] == cfg.tag_unlabelled].copy()

    # get list with numbers, how many samples to take in each round
    nr_samples_to_take = np.full(adapt_rounds, nr_samp_method // adapt_rounds)
    nr_samples_to_take[:nr_samp_method // adapt_rounds] += 1

    # iterate over list
    for nr_samples in nr_samples_to_take:
        # create training data
        x_train = x_data[df_validation.index]
        y_train = prediction_result[df_validation.index]

        # shuffle data
        shuffled_indices = np.random.permutation(len(x_train))
        x_train = x_train[shuffled_indices]
        y_train = y_train[shuffled_indices]

        # convert data to tf inputs
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

        # train model
        train_model(model, x_train, y_train)

        # get predictions of the model
        df_unlabelled_indices = df_unlabelled.index
        x_unlabelled = x_data[df_unlabelled_indices]
        x_unlabelled = tf.convert_to_tensor(x_unlabelled, dtype=tf.float32)
        y_unlabelled = model(x_unlabelled)
        y_unlabelled = tf.squeeze(y_unlabelled)

        # find the indices of the n lowest predictions
        lowest_indices = tf.argsort(y_unlabelled)[:nr_samples]

        # find the corresponding indices in the df unlabelled
        indices_to_annotate = df_unlabelled_indices[lowest_indices]

        # add samples to training set
        df.loc[indices_to_annotate, tag_col] = cfg.tag_train

        # for next iteration, move the sampled data from df2_unlabelled to df2_validation
        rows_to_move = df_unlabelled.loc[indices_to_annotate]
        df_validation = pd.concat([df_validation, rows_to_move])
        df_unlabelled.drop(indices_to_annotate, inplace=True)

        # for next iteration, prediction value is correct for new samples
        prediction_result[indices_to_annotate] = 1

    return df


def sampling_transfer_diversity(df, iteration, model, nr_samp_method, x_data_original, adapt_rounds=5):
    # get tag_col and last tag_col
    tag_col = utils.get_tag_iteration_column(iteration)
    last_tag_col = utils.get_tag_iteration_column(iteration - 1)

    # Model preparation: freeze all layers except the last one
    for layer in model.layers[:-1]:
        layer.trainable = False
    # exchange the last layer to have only one node
    model.pop()
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # compile model
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    # initialize validation and unlabelled set (x)
    index_unlabelled = df[df[last_tag_col] == cfg.tag_unlabelled].index
    index_validation = df[df[last_tag_col] == cfg.tag_validation].index

    # initialize label
    label = np.zeros(len(df))
    label[index_validation] = 1

    # get list with numbers, how many samples to take in each round
    nr_samples_to_take = np.full(adapt_rounds, nr_samp_method // adapt_rounds)
    nr_samples_to_take[:nr_samp_method // adapt_rounds] += 1

    for nr_samples in nr_samples_to_take:
        # create training data
        index_training = index_unlabelled.union(index_validation)
        x_train = x_data_original[index_training]
        y_train = label[index_training]

        # convert data to tf inputs
        x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

        train_model(model, x_train, y_train)

        # get predictions
        x_unlabelled = x_data_original[index_unlabelled]
        x_unlabelled = tf.convert_to_tensor(x_unlabelled, dtype=tf.float32)
        y_unlabelled = model(x_unlabelled)
        y_unlabelled = tf.squeeze(y_unlabelled)

        # find the indices of the n lowest predictions
        lowest_indices = tf.argsort(y_unlabelled)[:nr_samples]

        # find the corresponding indices in the df unlabelled
        indices_to_annotate = index_unlabelled[lowest_indices]

        # add samples to training set
        df.loc[indices_to_annotate, tag_col] = cfg.tag_train

        # for the next iteration, move samples to annotate from unlabelled to validation
        index_validation = index_validation.union(indices_to_annotate)
        index_unlabelled = index_unlabelled.difference(indices_to_annotate)

        label[indices_to_annotate] = 1
    return df


'''
Combination methods
'''
def sampling_cluster_preselected_samples(df, nr_samp_method, tag_col, x_data_original, y_uncertainty):
    # apply max combination
    y_uncertainty = np.max(y_uncertainty, axis=1)

    # get df with only unlabelled data
    df_unlabelled = df[df[tag_col] == cfg.tag_unlabelled].copy()
    indices_unlabelled = df_unlabelled.index

    # calculate nr of preselected samples (50% of unlabelled data)
    nr_preselected = int(len(df_unlabelled) / 2)

    if nr_preselected <= nr_samp_method:
        df = sampling_random(df, nr_samp_method, tag_col)
    else:
        # use only unlabelled data
        y_unlabelled_uncertainty = y_uncertainty[indices_unlabelled]
        # use the 50% of unlabelled data with highest uncertainty score
        highest_indices = np.argsort(y_unlabelled_uncertainty)[-nr_preselected:]
        indices_unlabelled = indices_unlabelled[highest_indices]
        df_unlabelled = df_unlabelled.loc[indices_unlabelled]

        # apply clustering
        df_unlabelled = sampling_clustering(df_unlabelled, nr_samp_method, tag_col, x_data_original)

        # add to training data
        df_unlabelled_indices = df_unlabelled[df_unlabelled[tag_col] == cfg.tag_train].index
        df.loc[df_unlabelled_indices, tag_col] = cfg.tag_train
    return df


'''
Validation data generation
'''

def get_validation_data(df, iteration, y_true, validation_split=0.2):
    tag_col = utils.get_tag_iteration_column(iteration)

    # filter training rows (Tag df and y array)
    df_training = df[df[tag_col] == cfg.tag_train]
    y_true_training = y_true[df[tag_col] == cfg.tag_train]

    # try stratified sampling, if not possible do random sampling
    try:
        _, val_df = train_test_split(df_training,
                                     test_size=validation_split,
                                     stratify=y_true_training,
                                     random_state=1)
    except:
        val_df = df_training.sample(frac=validation_split)

    # change label from training to validation
    df.loc[val_df.index, tag_col] = cfg.tag_validation

    return df


def initial_training_data_devision():
    if os.path.exists(cfg.path_results_dataset_al / 'initial_training_set.pickle'):
        return

    # load metadata
    file_name = cfg.selected_dataset + '_metadata.pickle'
    df_metadata = pd.read_pickle(cfg.path_results_dataset_metadata / file_name)

    # get tag iteration col and only use that column
    tag_col = utils.get_tag_iteration_column(0)
    df_metadata = df_metadata[[tag_col]]

    #nr of initial training examples
    nr_init_train = cfg.selected_initial_training_samples

    # get indices of unlabelled data
    unlabelled_indices = df_metadata[df_metadata[tag_col] == cfg.tag_unlabelled].index

    # iterate over random seeds
    for random_seed in cfg.random_seed_value:
        # column name
        col_name = f'tag_seed_{random_seed}'

        # set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        # get random indices to change for training
        indices_training = np.random.choice(unlabelled_indices, nr_init_train, replace=False)

        # store column with initial training data
        df_metadata[col_name] = df_metadata[tag_col]
        df_metadata.loc[indices_training, col_name] = cfg.tag_train

    # save df
    df_metadata = df_metadata.drop(columns=tag_col)
    df_metadata.to_pickle(cfg.path_results_dataset_al / 'initial_training_set.pickle')


def train_model(model, x_train, y_train):
    # Define the EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  # Metric to monitor for early stopping ('val_loss' or 'val_accuracy', etc.)
        patience=5,  # Number of epochs with no improvement after which training will be stopped
        restore_best_weights=True,  # Restores the model weights from the epoch with the best validation performance
        min_delta=0.1,  # minimum improvement
        mode='auto',  # stop when accuracy stops increasing
        verbose=0
    )
    model.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[early_stopping], verbose=0)

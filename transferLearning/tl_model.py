import os
import sys
from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage.transform import resize

import config as cfg
import utils
import utils_spectrogram

package_dir = Path(__file__).resolve().parent / 'models' / 'vggish'
sys.path.append(str(package_dir))
from transferLearning.models.vggish import vggish_input, vggish_slim, vggish_params


def calculate_and_save_embeddings():
    # create folder in not already created
    utils.create_folder(cfg.path_results_dataset_tl_embedding)

    # load metadata of selected dataset
    df_metadata = pd.read_pickle(cfg.path_file_metadata)

    # iterate over all embedding models
    for selected_embedding in cfg.selected_embeddings:
        embedding_model = selected_embedding[0].split('_')[0]

        if embedding_model == 'birdnet':
            birdnet_calculate_and_save_embeddings(df_metadata)
        elif embedding_model == 'yamnet':
            yamnet_calculate_embeddings(df_metadata)
        elif embedding_model == 'vgg16':
            vgg16_calculate_embeddings(df_metadata)
        elif embedding_model == 'vggish':
            vggish_calculate_embeddings(df_metadata)
        elif embedding_model == 'resnet152v2':
            resnet152v2_0_calculate_embeddings(df_metadata)


def embeddings_exist(model_name):
    embedding_list = [item for sublist in cfg.selected_embeddings for item in sublist if item.startswith(model_name)]
    for embedding in embedding_list:
        embedding_file = embedding + '.npy'
        if not os.path.exists(cfg.path_results_dataset_tl_embedding / embedding_file):
            return False
    return True


###########
# BirdNET #
###########
def birdnet_calculate_and_save_embeddings(df):

    def save_birdnet_embedding(birdnet_embedding, dimension, name_to_save):
        birdnet_embedding = tf.stack(birdnet_embedding, axis=0)
        birdnet_embedding = birdnet_embedding.numpy()
        birdnet_embedding = birdnet_embedding.reshape(len(df), dimension)
        np.save(cfg.path_results_dataset_tl_embedding / name_to_save, birdnet_embedding)

    if embeddings_exist('birdnet'):
        return

    print('SAVE BIRDNET EMBEDDINGS')

    # load model
    original_model = tf.keras.models.load_model(cfg.path_birdnet_pb).model
    layer_names = ['GLOBAL_AVG_POOL', 'POST_CONV_1', 'BLOCK_4-4_ADD']
    outputs = [original_model.get_layer(name).output for name in layer_names]
    model = tf.keras.models.Model(inputs=original_model.input, outputs=outputs)

    # initialize output
    birdnet_1 = [tf.zeros((1, 1024))] * len(df)
    birdnet_2 = [tf.zeros((1, 1, 6, 1024))] * len(df)
    birdnet_3 = [tf.zeros((1, 3, 8, 192))] * len(df)

    with tqdm(total=len(df)) as pbar:
        for index, row in df.iterrows():

            # get audio data
            audio_data = utils.load_audio_file(row['audio_path'], embedding='birdnet')
            audio_data = np.expand_dims(audio_data, axis=0)

            # get embeddings
            embeddings = model(audio_data)

            # store embeddings
            birdnet_1[index] = embeddings[0]
            birdnet_2[index] = embeddings[1]
            birdnet_3[index] = embeddings[2]

            pbar.update(1)

        # save embeddings
        save_birdnet_embedding(birdnet_1, 1024, 'birdnet_1.npy')
        save_birdnet_embedding(birdnet_2, 1 * 6 * 1024, 'birdnet_2.npy')
        save_birdnet_embedding(birdnet_3, 1 * 3 * 8 * 192, 'birdnet_3.npy')


##########
# YAMNet #
##########
def yamnet_calculate_embeddings(df):

    if embeddings_exist('yamnet'):
        return

    print('SAVE YAMNET EMBEDDINGS')

    # load yamnet model
    model = hub.load(cfg.url_yamnet)

    # initialize output
    yamnet_1 = [tf.zeros((6, 1024))] * len(df)

    # create and save embeddings
    with tqdm(total=len(df)) as pbar:
        for index, row in df.iterrows():

            # get audio data
            audio_data = utils.load_audio_file(row['audio_path'], embedding='yamnet')

            # get embeddings
            _, embedding, _ = model(audio_data)

            # store embeddings
            yamnet_1[index] = embedding

            pbar.update(1)

        # save embeddings
        yamnet_1 = [tensor.numpy() for tensor in yamnet_1]
        name_to_save = f'yamnet_1.npy'
        np.save(cfg.path_results_dataset_tl_embedding / name_to_save, yamnet_1)


#########
# VGG16 #
#########
def vgg16_calculate_embeddings(df):
    def save_vgg16_embeddings(vgg16_x, name_to_save):
        vgg16_x = [element[0] for element in vgg16_x]
        vgg16_x = np.array(vgg16_x)
        np.save(cfg.path_results_dataset_tl_embedding / name_to_save, vgg16_x)

    if embeddings_exist('vgg16'):
        return

    print('SAVE VGG16 EMBEDDING')

    # load Vgg16 model
    input_shape = (224, 224, 3)
    original_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)

    layer_names = ['predictions', 'fc2', 'fc1', 'flatten']
    outputs = [original_model.get_layer(name).output for name in layer_names]
    model = tf.keras.models.Model(inputs=original_model.input, outputs=outputs)

    # create new cols in df
    vgg16_1 = [tf.zeros((1, 4096))] * len(df)
    vgg16_2 = [tf.zeros((1, 4096))] * len(df)
    vgg16_3 = [tf.zeros((1, 25088))] * len(df)

    # create and save embeddings
    with tqdm(total=len(df)) as pbar:
        for index, row in df.iterrows():

            spectrogram = utils_spectrogram.get_spectrogram_image(row['audio_path'])
            spectrogram = resize(spectrogram, input_shape, mode='constant')
            spectrogram_batch = np.expand_dims(spectrogram, axis=0)
            embedding = model(spectrogram_batch)

            vgg16_1[index] = embedding[1]
            vgg16_2[index] = embedding[2]
            vgg16_3[index] = embedding[3]

            pbar.update(1)

        # save embeddings
        save_vgg16_embeddings(vgg16_1, 'vgg16_1.npy')
        save_vgg16_embeddings(vgg16_2, 'vgg16_2.npy')
        save_vgg16_embeddings(vgg16_3, 'vgg16_3.npy')


##########
# VGGish #
##########
def vggish_calculate_embeddings(df):

    if embeddings_exist('vggish'):
        return

    print('SAVE VGGISH EMBEDDINGS')

    # load model and compute embeddings
    with tf.compat.v1.Graph().as_default(), tf.compat.v1.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, cfg.path_checkpoint_vggish)
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        # create new col in df
        vggish_0 = [tf.zeros((3, 128))] * len(df)

        # create and save embeddings
        with tqdm(total=len(df)) as pbar:
            for index, row in df.iterrows():
                file_batch = vggish_input.wavfile_to_examples(row['audio_path'])
                [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: file_batch})

                vggish_0[index] = embedding_batch

                pbar.update(1)

            # save embeddings
            vggish_0 = np.array(vggish_0)
            np.save(cfg.path_results_dataset_tl_embedding / 'vggish_0.npy', vggish_0)


###############
# Resnet152v2 #
###############
def resnet152v2_0_calculate_embeddings(df):

    if embeddings_exist('resnet152v2'):
        return

    print('SAVE RESNET152V2 EMBEDDINGS')

    # define input shape
    input_shape = (224, 224, 3)

    # load resnet152v2 model
    original_model = tf.keras.applications.resnet_v2.ResNet152V2(include_top=True, weights='imagenet')
    layer_names = ['predictions', 'avg_pool']
    outputs = [original_model.get_layer(name).output for name in layer_names]
    model = tf.keras.models.Model(inputs=original_model.input, outputs=outputs)

    resnet152v2_1 = [tf.zeros((1, 2048))] * len(df)

    # create and save embeddings
    with tqdm(total=len(df)) as pbar:
        for index, row in df.iterrows():

            # claculate spectrogram and preprocess input
            spectrogram = utils_spectrogram.get_spectrogram_image(row['audio_path'])
            spectrogram = resize(spectrogram, input_shape, mode='constant')
            spectrogram = np.expand_dims(spectrogram, axis=0)
            spectrogram = tf.keras.applications.resnet_v2.preprocess_input(spectrogram)

            # calculate embeddings
            embedding = model(spectrogram)

            # save results
            resnet152v2_1[index] = embedding[1]

            pbar.update(1)

        # save embeddings
        resnet152v2_1 = [element[0] for element in resnet152v2_1]
        resnet152v2_1 = np.array(resnet152v2_1)
        np.save(cfg.path_results_dataset_tl_embedding / 'resnet152v2_1.npy', resnet152v2_1)

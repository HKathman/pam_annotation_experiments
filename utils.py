import os

import librosa
import numpy as np
import pandas as pd

import config as cfg


##################
# Create Folders #
##################
def create_folder(path):
    os.makedirs(path, exist_ok=True)


################################
# Extract Y from metadata file #
################################
def get_y_original_from_metadata():
    file_metadata = cfg.selected_dataset + '_metadata.pickle'
    df_metadata = pd.read_pickle(cfg.path_results_dataset_metadata / file_metadata)
    label_column = df_metadata.columns[df_metadata.columns.str.contains(r'\[|\]')].tolist()
    y_data_original = np.squeeze(np.array(df_metadata[label_column].values.tolist()))
    if y_data_original.ndim == 1:
        y_data_original = y_data_original.reshape(-1, 1)
    return y_data_original


####################
# Audio Processing #
####################
def get_audio_file_paths(directory):
    audio_extensions = ['.mp3', '.wav', '.flac']  # Add more extensions if needed
    audio_file_paths = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                audio_file_paths.append(os.path.join(root, file))

    return audio_file_paths


def load_audio_file(path, embedding='birdnet', offset=0.0, duration=None):
    if embedding == 'birdnet':
        signal, sample_rate = librosa.load(path, sr=cfg.birdnet_sample_rate, offset=offset,
                                           duration=duration, mono=True, res_type="kaiser_fast")

        # adjust duration for birdnet input
        if len(signal) > cfg.birdnet_sample_length:
            start = int((len(signal) - cfg.birdnet_sample_length) / 2)
            end = start + cfg.birdnet_sample_length
            signal = signal[start:end]
        elif len(signal) < cfg.birdnet_sample_length:
            padding = np.random.normal(0, max(abs(signal)/4), cfg.birdnet_sample_length - len(signal))
            signal = np.concatenate((padding[:int(len(padding)/2)], signal, padding[int(len(padding)/2):]))
    elif embedding == 'yamnet':
        signal, sample_rate = librosa.load(path, sr=cfg.yamnet_sample_rate, offset=offset,
                                           duration=duration, mono=True, res_type="kaiser_fast")
        signal = signal / (1.01*max(abs(signal)))
        signal = signal.astype(np.float32)
    return signal


#############################
# Active Learning Functions #
#############################
def get_tag_iteration_column(iteration=''):
    return 'tag_iteration_' + str(iteration)


def get_prediction_iteration_column(iteration=''):
    return 'prediction_iteration_' + str(iteration)


def create_experiment_file_name(data, emb, layer, hidden_layer, sampl, ending=True):
    return data.lower() + '_' + emb.lower() + '_' + str(layer) + '_' + str(hidden_layer) + '_' + sampl + '.pickle'

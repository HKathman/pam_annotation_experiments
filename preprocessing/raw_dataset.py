import os

import pandas as pd

import utils
import config as cfg


def format_and_save_annotations():
    # create folder in not already created
    utils.create_folder(cfg.path_results_dataset_metadata)

    # create file
    if os.path.exists(cfg.path_file_metadata):
        return
    else:
        print('SAVE METADATA\t| Dataset: ', cfg.selected_dataset, '\t| Path: ', cfg.path_file_metadata)
        if cfg.selected_dataset == 'example_dataset':
            df = create_example_dataset_metadata()

        df.to_pickle(cfg.path_file_metadata)


def create_example_dataset_metadata():
    df = pd.read_csv(cfg.path_example_dataset_metadata_original)

    # species to one vector, sorted by frequency
    species = [string for string in df.columns.tolist() if string.isupper()]
    occurrences = [df[col].sum() for col in species]
    sorted_species = [s for _, s in sorted(zip(occurrences, species), reverse=True)]
    df[str(sorted_species)] = df.apply(lambda row: [row[col] for col in sorted_species], axis=1)

    # get audio paths
    audio_paths = utils.getAudioFilePaths(cfg.path_example_dataset_audio)
    df['audio_key'] = df['fname'] + '_' + df['min_t'].astype(str) + '_' + df['max_t'].astype(str) + '.wav'
    audio_dict = {os.path.basename(path): path for path in audio_paths}
    df['audio_path'] = df['audio_key'].map(audio_dict)

    # create test set and unlabelled set
    df.loc[df['subset'] == 'test', 'subset'] = cfg.tag_test
    df.loc[df['subset'] == 'train', 'subset'] = cfg.tag_unlabelled

    # save only interesting columns, rename them
    columns = {'audio_path': 'audio_path',
               # 'fname': 'file',
               # 'min_t': 'start_time',
               # 'max_t': 'end_time',
               # 'date': 'date',
               # 'site': 'stratified_site',
               str(sorted_species): str(sorted_species),
               'subset': utils.getTagIterationColumn(0)}

    df = df[list(columns.keys())]
    df.rename(columns=columns, inplace=True)

    return df

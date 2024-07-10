from pathlib import Path

##############################
# variables selected by user #
##############################

# dataset
selected_dataset_int = 0

if selected_dataset_int == 0:
    selected_dataset = 'example_dataset'

# transfer learning, select embedding models | all possible models for embedding
selected_embeddings = [['birdnet_1', 'birdnet_2', 'birdnet_3'],
                       # ['yamnet_1'],
                       ['vggish_0'],
                       ['vgg16_1', 'vgg16_2', 'vgg16_3'],
                       ['resnet152v2_1']]

# active learning #
# selected embedding | use the embedding that performed best
selected_embedding = 'birdnet_1'

# nr of samples for initial training
selected_initial_training_samples = 20
selected_add_training_samples = 20
selected_max_training_samples = 1400

# selected uncertainty sampling strategies for average / max combination evaluation
selected_sampling_strategies = ['random_xx_100',

                                'leastconfidence_average_95',
                                'ratio_average_95', 'ratio_max_95',
                                'entropy_average_95',

                                'clustering_xx_95',

                                'transferUncertainty_xx_95',
                                'transferDiversity_xx_95',

                                'combineUncertaintyDiversity_xx_95',
                                'combineTransferUncertaintyDiversity_xx_95',
                                'combineUncertaintyTransferDiversity_xx_95',
                                'combineTransferUncertaintyTransferDiversity_xx_95']

#####################
# general variables #
#####################
# result folder
dir_results = 'results'
path_results = Path(__file__).parent /dir_results

# dataset path
path_dataset = path_results / selected_dataset

# metadata file / folder
path_results_dataset_metadata = path_dataset / '01-dataset_metadata'
file_metadata = selected_dataset + '_metadata.pickle'
path_file_metadata = path_results_dataset_metadata / file_metadata

# transfer learning path
path_results_dataset_tl = path_dataset / '02-transfer_learning'
path_results_dataset_tl_embedding = path_results_dataset_tl / 'embeddings'
path_results_dataset_tl_trained_models = path_results_dataset_tl / 'trained_models'
path_results_dataset_tl_trained_models_logs = path_results_dataset_tl_trained_models / 'logs'
path_results_dataset_tl_evaluation = path_results_dataset_tl / 'evaluation'

# active learning path
path_results_dataset_al = path_dataset / '03-active_learning'
path_results_dataset_al_experiments = path_results_dataset_al / 'experiments'
path_results_dataset_al_experiments_history = path_results_dataset_al_experiments / 'history'
path_results_dataset_al_experiments_logs = path_results_dataset_al_experiments / 'logs'
path_results_dataset_al_experiments_predictions = path_results_dataset_al_experiments / 'predictions'
path_results_dataset_al_evaluation = path_results_dataset_al / 'evaluation'

# figures path
dir_figures = 'figures'
path_figures = Path(__file__).parent / dir_figures

# seed for random functions
random_seed_value = list(range(6))

# dataset tags
tag_train = 'training'
tag_test = 'evaluation'
tag_unlabelled = 'unlabelled'
tag_validation = 'validation'

# result column tags
col_amountData = 'amount_data'


##################################
########## RAW DATASETS ##########
##################################
dir_data = 'data'

# example dataset variables
dir_example_dataset = 'example_dataset'
dir_example_dataset_audio = 'audio'
file_example_dataset_metadata = 'metadata.csv'
path_example_dataset_audio = Path(__file__).parent / dir_data / dir_example_dataset / dir_example_dataset_audio
path_example_dataset_metadata_original = Path(__file__).parent / dir_data / dir_example_dataset / file_example_dataset_metadata



##############################################
########## TRANSFER LEARNING MODELS ##########
##############################################

###########
# BirdNet #
###########
path_birdnet_pb = Path(__file__).parent / 'transferLearning' / 'models' / 'BirdNET-Analyzer-V2.4' / 'V2.4' \
                  / 'BirdNET_GLOBAL_6K_V2.4_Model'
birdnet_sample_rate = 48000 # 48kHz
birdnet_sample_length_sec = 3
birdnet_sample_length = birdnet_sample_rate * birdnet_sample_length_sec

##########
# YAMNet #
##########
url_yamnet = 'https://tfhub.dev/google/yamnet/1'
yamnet_sample_rate = 16000  # 16kHz


##########
# VGGish #
##########
path_checkpoint_vggish = str(Path(__file__).parent / 'transferLearning' / 'models' / 'vggish' / 'vggish_model.ckpt')






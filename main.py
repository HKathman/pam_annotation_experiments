import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

import config as cfg
from preprocessing import raw_dataset
from transferLearning import tl_model, tl_dataset_devision #tl_training as tlTrain, tl_evaluation as tlEval
#from activeLearning import al_training as alTrain, al_evaluation as alEval


# STEP 1: Unifying dataset format
raw_dataset.format_and_save_annotations()

# STEP 2: Create the embeddings of the dataset model
tl_model.calculate_and_save_embeddings()

# STEP 3: Create data devision dataframe (transfer learning without active learning)
tl_dataset_devision.create_data_tag_for_transfer_learning()

'''
STEP 4: Train and save models with transferLearning embeddings
'''
#tlTrain.trainAndSaveTransferLearningModels()

'''
STEP 5: Evaluate transfer learning models
'''
#for f1_metric in ['micro', 'macro']:
#    for data_tag in [cfg.tag_validation, cfg.tag_train, cfg.tag_test]:
#        tlEval.evaluateTransferLearningModels(data_tag, f1_metric)


'''
STEP 6: Embedding: 'birdnet_1' | Evaluate active learning uncertainty score generation (max vs. average)
'''
#alTrain.trainActiveLearningModels()


'''
STEP 7: Embedding: 'birdnet_1' | Create lists with variables used to create plots
'''
#alEval.evaluateActiveLearningModels()
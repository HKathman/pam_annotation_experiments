# pam_annotation_experiments
Pipeline to test different embedding models (Transfer Learning) and sampling strategies (Active Learning) to efficiently annotate Passive Acoustic Monitoring datasets

## Code Setup

### Set up virtual environment
```bash
C:\Python310\Python.exe -m venv venv
```

### Activate virtual environment
```bash
.\venv\Scripts\activate
```

### Install dependencies

```bash
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
```

### Install requirements

```bash
pip install -r requirements.txt 
```

# Data Setup
Audio of the datasets should be 3 seconds long.
A custom preprocessing function needs to be implemented individually.

# Pipeline Setup
### STEP 1: Unifying dataset format
1. Add the name of your dataset to selected_dataset in config.py 
(e.g. by adding: elif selected_dataset_int == 1: selected_dataset = 'my_awesome_dataset')
and change the selected_dataset_int value respectively. 
2. Add your custom unification function to raw_dataset.py 
(e.g. by adding: if cfg.selected_dataset == 'my_awesome_dataset': df = create_my_awesome_df())
Provide all constants (e.g. paths) in the config.py section RAW DATASETS.
Create a function in raw_dataset.py that outputs a pandas Dataframe in the same format as for the example 
dataset, with the 3 columns "audio_path", "['species_1', 'species_2', ...]" and "subset"
3. The dataframe will be stored at 
results/my_awesome_dataset/01-dataset_metadata/my_awesome_dataset_metadata.pickle

### STEP 2: Embed the dataset
1. Create a folder "models" in folder "transferLearning" and download birdnet v2.4 (https://github.com/kahst/BirdNET-Analyzer/tree/main/checkpoints)
and vggish (https://github.com/tensorflow/models/tree/master/research/audioset/vggish) into that folder.
Make sure that the file paths in config.py match your local file paths.

Using the embeddings specified in config.py, embeddings for different embedding models and layers are generated 
and stored in results/my_awesome_dataset/02-transfer_learning\embeddings\embeddingModel_embeddingLayer.npy
(This may take some time depending on the size of the dataset.)

### STEP 3: Create data devision dataframe (transfer learning without active learning)
Creates a data devision df for transfer learning with the different datasets used for training. Stored in
results/example_dataset/02-transfer_learning/training_split.pickle

### STEP 4: Transfer Learning Models
Creates the transfer learning models with the number of random seeds specified in config.py and saves the models to
results/example_dataset/02-transfer_learning/trained_models.




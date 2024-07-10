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
dataset, with the 2 columns 'audio_path' and "['species_1', 'species_2', ...]"
3. The dataframe will be stored at 
results/my_awesome_dataset/01-dataset_metadata/my_awesome_dataset_metadata.pickle
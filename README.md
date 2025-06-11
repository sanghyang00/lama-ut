# LAMA-UT: Language Agnostic Multilingual ASR through Orthography Unification and Language-Specific Transliteration
Official PyTorch Implementation of LAMA-UT: Language Agnostic Multilingual ASR through Orthography Unification and Language-Specific Transliteration (AAAI 2025 Oral Presentation)

Please note that the dataset itself is not included due to its large size. 
For the FLUERS dataset, the download code is integrated into the preprocessing script. 
For the CommonVoice (CV) dataset, you can download the compressed files for each language from CommonVoice's official website. (https://commonvoice.mozilla.org/en/datasets)
After downloading the datasets, please organize the folders according to the structure outlined in the "CV_example.txt" file.

The following sections describe each file and its respective arguments.
Files not detailed here are utilized primarily for declaring or importing functions and variables.

1. preprocess.py

    Arguments
    - DATASET: determine which dataset to preprocess between FLEURS and CommonVoice.
    - CV_PATH (optional): root directory in accordance with the structure specified in the CV_example.txt file. Only necessary in preprocessing the CV dataset.

    Returns
    - Preprocessed TSV files (train, dev, test split for FLEURS, train, test split for CV)

2. train.py

    Arguments
    - model_size: determine the size of the trained universal transcription generator.
    - batch: determine batch size during training.
    - minibatch: the size of the minibatch used in gradient accumulation.
    - total_step: total steps to train.
    - init_lr: initial learning rate of tri-stage learning rate scheduler.
    - peak_lr: peak learning rate of tri-stage learning rate scheduler.
    - final_lr: final learning rate of tri-stage learning rate scheduler.
    - data_dir: root directory which contains preprocessed TSV files. (FLEURS)

    Returns
    - Checkpoint file (.pt) of trained universal transcription generator

3. inference.py

    Arguments
    - TRAIN_DATA_PATH: path to train.tsv file of the desired dataset.
    - TEST_DATA_PATH: path to test.tsv file of the desired dataset.
    - VOCAB_PATH: path to vocabulary file path.
    - UTG_SIZE: the size of the universal transcription generator.
    - CKPT_PATH: path to checkpoint file (.pt) of the universal transcription generator.
    - DEVICE: a device used for inference.
    - OPENAI_API_KEY: OpenAI API key to leverage GPT API.
    - PROMPT_TYPE: types of prompts used for inference with the universal converter.

    Returns
    - Printed pair of ground truth transcription and predicted transcription

To reproduce the results, 
1. download the CommonVoice dataset, 
2. organize it according to the specified file structure, 
3. execute the following scripts in order: "preprocess.py", "train.py", and "inference.py".

import os
from datasets import Dataset
import soundfile as sf
import transformers
from transformers import Wav2Vec2CTCTokenizer, HubertConfig, HubertForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
# from transformers import AutoTokenizer, AutoFeatureExtractor, AutoProcessor
# import torchaudio
# import numpy as np
import torch
import traceback
import multiprocessing
import json
import gc

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_data_dev_test(root_dir, split_name):
    data = {"path": [], "transcript": [], "split": []}

    split_dir = os.path.join(root_dir, split_name)
    for category in os.listdir(split_dir):
        category_dir = os.path.join(split_dir, category)

        # For each file
        for file in os.listdir(category_dir):
            if file.endswith(".flac"):
                path = os.path.join(category_dir, file)
                transcript_path = path.replace(".flac", ".txt")

                with open(transcript_path, "r") as f:
                    transcript = f.read().strip()

                data["path"].append(path)
                data["transcript"].append(transcript)
                data["split"].append(split_name)

    return Dataset.from_dict(data)

def load_data_train(root_dir, train_split_name):
    data = {"path": [], "transcript": [], "split": []}

    split_dir = os.path.join(root_dir, train_split_name)

    for file in os.listdir(split_dir):
        if file.endswith(".flac"):
            path = os.path.join(split_dir, file)
            transcript_path = path.replace(".flac", ".txt")

            with open(transcript_path, "r") as f:
                transcript = f.read().strip()

            data["path"].append(path)
            data["transcript"].append(transcript)
            data["split"].append('Train')
    
    return Dataset.from_dict(data)

def map_to_array(batch):
    try:
        speech, _ = sf.read(batch["path"])
        batch["speech"] = speech
        return batch
    except Exception as e:
        print(f"Error in map_to_array: {e}")
        traceback.print_exc()


def prepare_dataset(batch):
    # Process the speech data
    # print("Running prepare_dataset function...")
    input_values = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding="longest").input_values#[0]
    # print(input_values.shape)

    if "transcript" in batch:
        # Process the transcripts
        with processor.as_target_processor():
            labels = processor(batch["transcript"], return_tensors="pt", padding="longest").input_ids#[0]

    # Combine the processed data into a single batch
    batch = {"input_values": input_values, "labels": labels}
    # print("Finished running prepare_dataset function.")
    return batch



# Load data
# dataset = load_data("/raid/kaisar_dauletbek/datasets/ISSAI_KSC2")

num_processes = 16  # Adjust this based on your system

# Initialize a new tokenizer
tokenizer = Wav2Vec2CTCTokenizer("/raid/kaisar_dauletbek/Kazakh-Hubert/vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token=" ")

# Initialize a new model configuration
config = HubertConfig()

# Initialize a new model with the configuration
# model = HubertModel(config)
model = HubertForCTC(config)

# Initialize a new feature extractor
feature_extractor = Wav2Vec2FeatureExtractor()

# Combine the tokenizer and feature extractor into a processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# try:
#     dataset = dataset.map(map_to_array, num_proc=num_processes)
#     print('speech to array done')
# except Exception as e:
#     print(f"Error in map_to_array map: {e}")
#     traceback.print_exc()

# dataset.save_to_disk("/raid/kaisar_dauletbek/Kazakh-Hubert/dataset_intermediate_step")

# try:
#     dataset = dataset.map(prepare_dataset, num_proc=num_processes)
#     print('prepare dataset done')
# except Exception as e:
#     print(f"Error in tokenize map: {e}")
#     traceback.print_exc()


# # save the dataset
# dataset.save_to_disk("/raid/kaisar_dauletbek/Kazakh-Hubert/dataset_proper_with_train")

# Load data and process it in three parts for training
train_splits = ['parliament', 'podcasts', 'radio', 'talkshow', 'tts', 'tv_news', 'crowdsourced']
for split in train_splits:
    dataset = load_data_train("/raid/kaisar_dauletbek/datasets/ISSAI_KSC2/Train", split)
    print(f"Loaded {split} data")
    dataset = dataset.map(map_to_array)
    dataset = dataset.map(prepare_dataset)#, num_proc=num_processes, batch_size=8, batched=True)
    print(f"Prepared {split} data")
    dataset.save_to_disk(f"/raid/kaisar_dauletbek/Kazakh-Hubert/dataset_main/{split.lower()}")
    del dataset
    gc.collect()



# Load and process the dev and test data
for split in ["Dev", "Test"]:
    dataset = load_data_dev_test("/raid/kaisar_dauletbek/datasets/ISSAI_KSC2", split)
    dataset = dataset.map(map_to_array)
    dataset = dataset.map(prepare_dataset)#, num_proc=num_processes, batch_size=8, batched=True)
    dataset.save_to_disk(f"/raid/kaisar_dauletbek/Kazakh-Hubert/dataset_main/{split.lower()}")
    del dataset
    gc.collect()




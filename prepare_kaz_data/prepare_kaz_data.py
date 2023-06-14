import os
from datasets import Dataset
import soundfile as sf
import transformers
from transformers import Wav2Vec2CTCTokenizer, HubertConfig, HubertForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoProcessor
import torchaudio
import numpy as np
import torch
import traceback
import multiprocessing
import json

def load_data(root_dir):
    data = {"path": [], "transcript": [], "split": []}

    # For each subdirectory
    # for split in ["Train", "Dev", "Test"]:
    for split in ['Dev', 'Test']:
        split_dir = os.path.join(root_dir, split)
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
                    data["split"].append(split)

    return Dataset.from_dict(data)

def tokenize(batch):
    try:
        tokens = tokenizer(batch["transcript"], padding="longest", truncation=True, max_length=512)
        return tokens
    except Exception as e:
        print(f"Error in tokenize: {e}")
        traceback.print_exc()

def map_to_array(batch):
    try:
        speech, _ = sf.read(batch["path"])
        batch["speech"] = speech
        return batch
    except Exception as e:
        print(f"Error in map_to_array: {e}")
        traceback.print_exc()

def prepare_dataset(batch):
    batch["input_values"] = processor(batch["speech"], sampling_rate=16000).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)

# Load data
dataset = load_data("/raid/kaisar_dauletbek/datasets/ISSAI_KSC2")

num_processes = 32  # Adjust this based on your system

# tokenizer = Wav2Vec2CTCTokenizer("/raid/kaisar_dauletbek/Kazakh-Hubert/vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token=" ")
model_checkpoint = "facebook/hubert-large-ls960-ft"
tokenizer = AutoTokenizer("/raid/kaisar_dauletbek/Kazakh-Hubert/vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token=" ")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
processor = AutoProcessor.from_pretrained(feature_extractor=feature_extractor, tokenizer=tokenizer)


try:
    dataset = dataset.map(tokenize, num_proc=num_processes)
    print('text tokenized')
except Exception as e:
    print(f"Error in tokenize map: {e}")
    traceback.print_exc()

try:
    dataset = dataset.map(map_to_array, num_proc=num_processes)
    print('speech to array done')
except Exception as e:
    print(f"Error in map_to_array map: {e}")
    traceback.print_exc()

# save the dataset
dataset.save_to_disk("/raid/kaisar_dauletbek/Kazakh-Hubert/dataset")

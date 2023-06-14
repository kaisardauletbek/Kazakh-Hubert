import os
from datasets import Dataset
import soundfile as sf
import transformers
from transformers import Wav2Vec2CTCTokenizer, HubertConfig, HubertForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
import torchaudio
import numpy as np
import torch
import traceback
from transformers import AutoProcessor
import multiprocessing
import json

def load_data(root_dir):
    data = {"path": [], "transcript": [], "split": []}

    # For each subdirectory
    for split in ["Train", "Dev", "Test"]:
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
    # Tokenize the transcripts and align the tokens with the original characters
    # with tokenizer.as_target_processor():
    tokens = tokenizer(batch["transcript"], padding="longest", truncation=True, max_length=512)
    return tokens

def map_to_array(batch):
    speech, _ = sf.read(batch["path"])
    batch["speech"] = speech
    return batch



# Initialize a new tokenizer
tokenizer = Wav2Vec2CTCTokenizer("/raid/kaisar_dauletbek/Kazakh-Hubert/vocab.json", unk_token='<unk>', pad_token='<pad>', word_delimiter_token=" ")

# Load data
dataset = load_data("/raid/kaisar_dauletbek/datasets/ISSAI_KSC2")

max_cpu = multiprocessing.cpu_count()
half_cpu = 64
dataset = dataset.map(tokenize, num_proc=half_cpu)
dataset = dataset.map(map_to_array, num_proc=half_cpu)

# save the dataset
dataset.save_to_disk("/raid/kaisar_dauletbek/Kazakh-Hubert/dataset")
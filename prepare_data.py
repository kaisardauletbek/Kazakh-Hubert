import torch
import torchaudio
import transformers
import soundfile as sf
import numpy as np
from torch.utils.data import DataLoader
from transformers import Wav2Vec2CTCTokenizer, HubertConfig, HubertForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, TrainingArguments, Trainer
from datasets import Dataset, load_from_disk, load_metric
from torch.nn.utils.rnn import pad_sequence
import multiprocessing

# Initialize a new tokenizer
tokenizer = Wav2Vec2CTCTokenizer("/raid/kaisar_dauletbek/Kazakh-Hubert/vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token=" ")

# Initialize a new feature extractor
feature_extractor = Wav2Vec2FeatureExtractor()

# Combine the tokenizer and feature extractor into a processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def prepare_dataset(batch):
    # Process the speech data
    print("Running prepare_dataset function...")
    input_values = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt", padding="longest").input_values#[0]
    print(input_values.shape)

    if "transcript" in batch:
        # Process the transcripts
        with processor.as_target_processor():
            labels = processor(batch["transcript"], return_tensors="pt", padding="longest").input_ids#[0]

    # Combine the processed data into a single batch
    batch = {"input_values": input_values, "labels": labels}
    print("Finished running prepare_dataset function.")
    return batch


num_proc = 8
dataset = load_from_disk('/raid/kaisar_dauletbek/Kazakh-Hubert/dataset')
dataset = dataset.map(prepare_dataset, remove_columns=["path", "transcript", "split"], num_proc=num_proc, batched=True, batch_size=16)
dataloader = DataLoader(dataset, shuffle=True, batch_size=16)

# save the dataset
dataset.save_to_disk("/raid/kaisar_dauletbek/Kazakh-Hubert/dataset_ready")

# Save the dataloaders for later usage
torch.save(dataloader, "/raid/kaisar_dauletbek/Kazakh-Hubert/dataloader.pt")
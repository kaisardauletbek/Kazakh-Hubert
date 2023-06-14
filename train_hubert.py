import torch
import torchaudio
import transformers
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2CTCTokenizer, HubertConfig, HubertForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
from torch.utils.data import DataLoader
import datasets
from datasets import Dataset, load_dataset
from transformers import TrainingArguments, Trainer

# Initialize a new tokenizer
tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token=" ")


# Train the tokenizer on your data
# tokenizer.train(files=dataset["path"])

# Initialize a new model configuration
config = HubertConfig()

# Initialize a new model with the configuration
model = HubertForCTC(config)

# Initialize a new feature extractor
feature_extractor = Wav2Vec2FeatureExtractor()

# Combine the tokenizer and feature extractor into a processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def prepare_dataset(batch):
    # check that all files have the correct sampling rate
    assert (
        len(set(batch["sampling_rate"])) == 1
    ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

    batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

    if "transcript" in batch:
        batch["labels"] = processor(batch["transcript"], padding="longest").input_ids
    return batch

num_proc = 64
dataset = load_dataset('dataset/ds.hf', split='train', cache_dir='dataset/cache')
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=64)
dataloader = DataLoader(dataset, shuffle=True, batch_size=16)



training_args = TrainingArguments(
    output_dir="/raid/kaisar_dauletbek/Kazakh-Hubert",
    group_by_length=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    learning_rate=1e-4,
    warmup_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=dataset,
)

trainer.train()


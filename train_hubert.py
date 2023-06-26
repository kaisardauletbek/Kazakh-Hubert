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


# Train the tokenizer on your data
# tokenizer.train(files=dataset["path"])

# Initialize a new model configuration
config = HubertConfig()

# Initialize a new model with the configuration
# model = HubertModel(config)
model = HubertForCTC(config)

# Initialize a new feature extractor
feature_extractor = Wav2Vec2FeatureExtractor()

# Combine the tokenizer and feature extractor into a processor
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


# def data_collator(features):
#     # Pad the input values
#     input_values = [feature["input_values"] for feature in features]
#     input_values = pad_sequence(torch.tensor(input_values), batch_first=True)

#     # Pad the labels
#     labels = [feature.get("labels", torch.tensor([-100])) for feature in features]
#     labels = pad_sequence(torch.tensor(labels), batch_first=True)

#     # Return the padded batch
#     return {"input_values": input_values, "labels": labels}

def data_collator(features):
    # Pad the input values
    input_values = [feature["input_values"] for feature in features]
    input_values = pad_sequence([torch.cat(input_value, dim=0) for input_value in input_values], batch_first=True)

    # Pad the labels
    labels = [feature.get("labels", torch.tensor([-100])) for feature in features]
    labels = pad_sequence([torch.cat(label, dim=0) for label in labels], batch_first=True)

    # Return the padded batch
    return {"input_values": input_values, "labels": labels}


data_collator = data_collator
wer_metric = load_metric("wer")

# Define the compute metrics function
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# dataloader = torch.load("/raid/kaisar_dauletbek/Kazakh-Hubert/dataloader.pt")
dataset = load_from_disk('/raid/kaisar_dauletbek/Kazakh-Hubert/dataset_ready')

# Define the data collator
# data_collator = DataCollatorCTC(processor, padding=True)

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

print("Start training")
trainer.train()


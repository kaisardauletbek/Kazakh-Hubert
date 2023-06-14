import torch
import torchaudio
import transformers
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2CTCTokenizer, HubertConfig, HubertForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Processor#, DataCollatorCTC
from torch.utils.data import DataLoader
import datasets
from datasets import Dataset, load_dataset, load_from_disk
from transformers import TrainingArguments, Trainer
from torch.nn.utils.rnn import pad_sequence

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

def prepare_dataset(batch):
    # Process the speech data
    print('HUI SPEECH')
    input_values = processor(batch["speech"], sampling_rate=16_000, return_tensors="pt").input_values
    print('HUI SPEECH DONE')

    if "transcript" in batch:
        # Process the transcripts
        with processor.as_target_processor():
            print('HUI TRANSCRIPT')
            labels = processor(batch["transcript"], return_tensors="pt").input_ids
            print('HUI TRANSCRIPT DONE')

    # Combine the processed data into a single batch
    batch = {"input_values": input_values, "labels": labels}
    return batch

def data_collator(features):
    # Pad the input values
    input_values = [feature["input_values"] for feature in features]
    input_values = pad_sequence(input_values, batch_first=True)

    # Pad the labels
    labels = [feature.get("labels", torch.tensor([-100])) for feature in features]
    labels = pad_sequence(labels, batch_first=True)

    # Return the padded batch
    return {"input_values": input_values, "labels": labels}


num_proc = 32
dataset = load_from_disk('/raid/kaisar_dauletbek/Kazakh-Hubert/dataset')
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names, num_proc=num_proc)
dataloader = DataLoader(dataset, shuffle=True, batch_size=16)

# Define the data collator
# data_collator = DataCollatorCTC(processor, padding=True)

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


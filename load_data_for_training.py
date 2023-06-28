import os
from datasets import Dataset, load_from_disk, concatenate_datasets
import glob

def load_split(split_dir):
    dataset = load_from_disk(split_dir)
    return dataset

def concatenate_train_splits(train_splits_dir):
    train_splits = glob.glob(os.path.join(train_splits_dir, '*'))
    train_splits.sort()
    train_dataset = load_split(train_splits[0])
    for split in train_splits[1:]:
        train_dataset = concatenate_datasets([train_dataset, load_split(split)])
    return train_dataset

print(concatenate_train_splits("/raid/kaisar_dauletbek/Kazakh-Hubert/dataset_main/train"))

print(load_split("/raid/kaisar_dauletbek/Kazakh-Hubert/dataset_main/dev"))

print(load_split("/raid/kaisar_dauletbek/Kazakh-Hubert/dataset_main/test"))
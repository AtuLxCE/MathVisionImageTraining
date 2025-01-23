# data/dataset.py

from .data_preparation import load_image_datasets, convert_to_conversation

def prepare_datasets():
    image_train_dataset, image_val_dataset = load_image_datasets()
    converted_image_train_dataset = [convert_to_conversation(sample) for sample in image_train_dataset]
    converted_image_val_dataset = [convert_to_conversation(sample) for sample in image_val_dataset]
    return converted_image_train_dataset, converted_image_val_dataset
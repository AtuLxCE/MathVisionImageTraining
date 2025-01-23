# data/data_preparation.py

from datasets import load_dataset

def load_image_datasets():
    image_train_dataset = load_dataset("Yugratna/geometric_image_data_v2", split="train")
    image_val_dataset = load_dataset("Yugratna/geometric_image_data_v2", split="validation")
    return image_train_dataset, image_val_dataset

def convert_to_conversation(sample):
    instruction = f"You are an expert geometric problem solver. Accurately describe the contents of the image and solve the problem presented: {sample['question']}."
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : sample["image_path"]} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["solution"]} ]
        },
    ]
    return { "messages" : conversation }
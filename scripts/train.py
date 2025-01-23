# scripts/train.py

from models.model import load_model, setup_lora
from models.trainer import setup_trainer
from data.dataset import prepare_datasets
from unsloth import unsloth_train

def train():
    image_model, image_tokenizer = load_model()
    image_model = setup_lora(image_model)
    train_dataset, val_dataset = prepare_datasets()
    trainer = setup_trainer(image_model, image_tokenizer, train_dataset, val_dataset)
    trainer_status = unsloth_train(trainer)
    return trainer_status

if __name__ == "__main__":
    train()
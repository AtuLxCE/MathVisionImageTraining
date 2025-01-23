# models/model.py

from unsloth import FastLanguageModel, FastVisionModel
from config.config import MODEL_NAME, MAX_SEQ_LENGTH, DTYPE, LOAD_IN_4BIT

def load_model():
    image_model, image_tokenizer = FastVisionModel.from_pretrained(
        MODEL_NAME,
        load_in_4bit=LOAD_IN_4BIT,
        use_gradient_checkpointing="unsloth",
    )
    return image_model, image_tokenizer

def setup_lora(image_model):
    image_model = FastVisionModel.get_peft_model(
        image_model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    return image_model
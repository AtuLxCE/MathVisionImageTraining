# models/trainer.py

from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from data.dataset import prepare_datasets


def setup_trainer(image_model, image_tokenizer, train_dataset, val_dataset):
    trainer = SFTTrainer(
        model=image_model,
        tokenizer=image_tokenizer,
        data_collator=UnslothVisionDataCollator(image_model, image_tokenizer),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=10,
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="output/",
            report_to="wandb",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=4,
            max_seq_length=2048,
            save_strategy="steps",
            save_steps=5,
        ),
    )
    return trainer
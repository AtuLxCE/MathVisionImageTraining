# scripts/inference.py

from models.model import load_model
from data.dataset import prepare_datasets
from transformers import TextStreamer

def inference():
    image_model, image_tokenizer = load_model()
    _, val_dataset = prepare_datasets()
    
    FastVisionModel.for_inference(image_model)
    
    image = val_dataset[0]["image_path"]
    instruction = f"You are an expert geometric problem solver. Accurately describe the contents of the image and solve the problem presented: {val_dataset[0]['question']}."
    
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    input_text = image_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = image_tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    
    text_streamer = TextStreamer(image_tokenizer, skip_prompt=True)
    _ = image_model.generate(**inputs, streamer=text_streamer, max_new_tokens=800,
                             use_cache=True, temperature=1.5, min_p=0.1)

if __name__ == "__main__":
    inference()
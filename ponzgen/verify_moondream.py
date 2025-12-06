
import sys
import os
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create a dummy image
img = Image.new('RGB', (100, 100), color = 'red')
img.save("test_moondream.png")

print("Loading Moondream2...")
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True, 
        revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    print("Model loaded. Encoding image...")
    image = Image.open("test_moondream.png")
    enc_image = model.encode_image(image)
    
    print("Image encoded. Answering question...")
    answer = model.answer_question(enc_image, "Describe this image.", tokenizer)
    print(f"Answer: {answer}")
    print("Verification SUCCESS")

except Exception as e:
    print(f"Verification FAILED: {e}")
    import traceback
    traceback.print_exc()

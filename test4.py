from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-32")

# Load model - this will be HUGE (45GB+)
model = AutoModelForCausalLM.from_pretrained(
    "google/switch-base-32",
    torch_dtype=torch.bfloat16,  # Use BF16 to save memory
    device_map="auto",  # Automatically distribute across available GPUs
    load_in_8bit=True,  # Optional: further reduce memory usage
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize input
inputs = tokenizer("The meaning of life is", 
                  return_tensors="pt", padding=True)

start=time.time()
# Generate
output = model.generate(
    inputs["input_ids"].to("cuda"), 
    attention_mask=inputs["attention_mask"].to("cuda"),
    do_sample=True,
    max_length=1000,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)
end=time.time()

# Decode and print
print(tokenizer.decode(output[0], skip_special_tokens=True))
print(end-start)
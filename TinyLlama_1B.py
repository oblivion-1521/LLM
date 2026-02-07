from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import random

def get_random_words(file_path, count):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        return random.sample(words, min(count, len(words)))
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []

# 1. Point to your local directory
model_path = "./Models/Llama_1B"

print("Loading model and tokenizer...")
start_time = time.time()

# 2. Load Tokenizer and Model
# Note: Since you have no GPU, we force CPU and use float32
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32, 
    device_map="cpu"
)

print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

# 3. Prepare the Input (Task: Words to Sentence)
words = ['sky', 'blue', 'sun'] 
# words = get_random_words("temp2.txt", random.randint(3, 5))
prompt = f"<|system|> You are a helpful assistant.</s>\n\
    <|user|>\nCreate a very short sentence using only these words: {words}.\
No extra adjectives. No extra nouns. \
Use simple present tense. Output one sentence only.</s>\n\
        <|assistant|>\n"
print(f"\nPrompt:\n{prompt}")
# 4. Tokenization
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

# 5. Generation(Inference)
print("Generating...")
output_tokens = model.generate(
    **inputs, 
    max_new_tokens=50,
    do_sample=False,      # Enable creative sampling
    # temperature=0.7,     # Randomness control
    # top_p=0.9            # Diversity control
)

# 6. Decoding
result = tokenizer.decode(output_tokens[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
print("\n--- Result ---")
print(result)
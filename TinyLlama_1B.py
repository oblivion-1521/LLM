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
# Use GPU if available (4090D), fall back to CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
# Speed tweak: use fp16 on CUDA to reduce memory bandwidth and increase throughput.
# 半精浮点，显存减半，且Tensor Core对FP16的计算速度是FP32的数倍
dtype = torch.float16 if device == "cuda" else torch.float32
# Speed tweak: enable TF32 on Ampere+ (4090) for faster matmul in any fp32 ops.
if device == "cuda":
    # Ampere架构抵用TensorFloat-32，降低极少精度来获得相似性能
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# 加载model和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=dtype,
    device_map=device
)
model.eval()  # Speed tweak: disable dropout and enable inference-optimized code paths.

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
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 5. Generation(Inference)
print("Generating...")

# Speed tweak: inference_mode avoids autograd overhead and reduces memory use.
# 告诉Torch是一个纯推理过程，不需要构建computation graph, 
                                # 不保存中间激活值，节省显存和CPU调度开销
with torch.inference_mode():
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,      # Enable creative sampling
        # temperature=0.7,     # Randomness control
        # top_p=0.9            # Diversity control
        use_cache=True        # Speed tweak: KV cache speeds up autoregressive decoding.
                                # KV Cache在生成低N个token的时候不需要重新计算前N-1个token的Attention Key，
                                # 而是直接从缓存中读取，大幅提升生成速度和减少显存使用
    )

# 6. Decoding
result = tokenizer.decode(output_tokens[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
print("\n--- Result ---")
print(result)
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
model_path = "./Models/LlaMa_8B"

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
    device_map="auto" if device == "cuda" else "cpu"
)
model.eval()  # Speed tweak: disable dropout and enable inference-optimized code paths.

# Avoid warning about missing pad token when using open-end generation.
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

# 3. 准备输入 (整合了你原有的长处：明确的指令 + Few-shot 样例)
words = ['sky', 'bIue', 'sun'] # 故意混入拼写错误

# 构造符合 Llama 3.1 标准的对话结构
messages = [
    {
        "role": "system", 
        "content": "You are a helpful assistant for a bio-sensor system. "
                   "Your task: 1. Correct spelling errors in the provided words (e.g., 'bospital' -> 'hospital'). "
                   "2. Create a short, natural sentence using those words, don't add irrelavant words, don't change the order of them. No extra nouns. No extra abjectives. The shorter the better. "
                   "3. Output exactly two lines: 'Corrected words: [list]' and 'Sentence: [sentence]'. "
                   "If no errors are found, keep the original words."
    },
    # 第一个例子 (Few-shot)
    {"role": "user", "content": "Words: ['eht', 'appie']"},
    {"role": "assistant", "content": "Corrected words: ['eat', 'apple']\nSentence: I eat an apple."},
    # 第二个例子 (Few-shot)
    {"role": "user", "content": "Words: ['whlk', 'sohool']"},
    {"role": "assistant", "content": "Corrected words: ['walk', 'school']\nSentence: I walk to school."},
    # 实际输入
    {"role": "user", "content": f"Words: {words}"}
]

# 使用官方模板转换
# add_generation_prompt=True 会在末尾添加 <|start_header_id|>assistant<|end_header_id|>\n\n
# 强制模型进入“回答阶段”
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 4. Tokenization
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 5. Generation
print("Generating...")

with torch.inference_mode():
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False, 
        use_cache=True,
        # 重点：Llama 3.1 必须指定 eos_token_id 为 <|eot_id|>
        eos_token_id=[
            tokenizer.eos_token_id, 
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ],
        pad_token_id=tokenizer.eos_token_id
    )

# 6. Decoding
# 截断 prompt 部分，只看生成的回答
result = tokenizer.decode(output_tokens[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
print("\n--- Result ---")
print(result)
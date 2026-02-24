# LlaMa_8B_interactive.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

# 1. 配置路径
model_path = "./Models/LlaMa_8B"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

if device == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

print("Loading model and tokenizer (safetensors + mmap)...")
start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 核心改进：丢掉 device_map，使用 safetensors 和 mmap 机制
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=dtype,
    use_safetensors=True,       # 强制使用 safetensors
    low_cpu_mem_usage=True,     # 配合 safetensors 触发底层 mmap 加载, 在使用 
    device_map={"": device}     # accelerate流式加载到显存，同时避免device_map='auto'可能带来的设备误切分问题
)

# 手动推入显存，获得纯净的 PyTorch 模型对象（无 hook 负担）, device_map=None的时候才需要这一行
# model.to(device)
model.eval()

print(f"Model loaded and moved to {device} in {time.time() - start_time:.2f} seconds.")

def generate_correction(words_input: str) -> str:
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant for a bio-sensor system. "
                "Your task: 1. Correct spelling errors in the provided words (e.g., 'bospital' -> 'hospital'). "
                "2. Create a short, natural sentence using those words. Don't add irrelavant words. Don't change the order of the words! Don't change the order of the words! No extra nouns. No extra abjectives. "
                "3. Output exactly two lines: 'Corrected words: [list]' and 'Sentence: [sentence]'. "
                "If no errors are found, keep the original words."},
        {"role": "user", "content": "Words: ['eht', 'appie']"},
        {"role": "assistant", "content": "Corrected words: ['eat', 'apple']\nSentence: I eat an apple."},
        {"role": "user", "content": "Words: ['whlk', 'sohool']"},
        {"role": "assistant", "content": "Corrected words: ['walk', 'school']\nSentence: I walk to school."},
        {"role": "user", "content": "Words: ['sky', 'bIue', 'sun']"},
        {"role": "assistant", "content": "Corrected words: ['sky', 'blue', 'sun']\nSentence: The sky is blue and the sun shines."},
        {"role": "user", "content": f"Words: {words_input}"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False, 
            temperature=None,
            top_p=None,
            use_cache=True,
            eos_token_id=[
                tokenizer.eos_token_id, 
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ],
            pad_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(output_tokens[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return result

# 3. 交互式循环 (常驻显存)
print("\n" + "="*50)
print("Biosensor Assistant Ready! (Type 'quit' or 'exit' to stop)")
print("="*50)

while True:
    try:
        user_input = input("\nEnter words list (e.g. ['sky', 'bIue', 'sun']): ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Exiting...")
            break
            
        if not user_input:
            continue
            
        t0 = time.time()
        res = generate_correction(user_input)
        t1 = time.time()
        
        print(f"\n--- Result ({t1-t0:.3f}s) ---")
        print(res)

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {e}")
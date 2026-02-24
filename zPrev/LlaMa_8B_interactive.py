# LlaMa_8B_interactive.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import os

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
    use_safetensors=True,
    low_cpu_mem_usage=True,
    device_map={"": device}
)

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

# 3. 强化版文件轮询监控 (支持多行独立处理、修改历史记录、缓存加速)
input_file = "input.txt"

if not os.path.exists(input_file):
    with open(input_file, "w", encoding="utf-8") as f:
        f.write("")

print("\n" + "="*50)
print(f"Biosensor Assistant Ready! Monitoring '{input_file}'...")
print("Press Ctrl+C to stop.")
print("="*50)

# 用于存储上一次解析出来的有效单词任务列表
last_parsed_tasks = []
# 用于缓存已经生成过的句子，避免修改第三行时，第一行也要重新生成一遍
llm_cache = {}

while True:
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 1. 保留换行符按行分割
        lines = content.splitlines(keepends=True)
        current_parsed_tasks = []
        
        # 2. 独立解析每一行
        for line in lines:
            if line.endswith('\n'):
                # 情况A：行末有换行符。说明这已经是完整的一句话（比如历史记录），提取其中的所有单词
                words = line.strip().split()
            else:
                # 情况B：行末无换行符。这通常是正在输入的最末行，严格以最后一个空格为界截断
                last_space_idx = line.rfind(' ')
                if last_space_idx != -1:
                    words = line[:last_space_idx].strip().split()
                else:
                    words = []
            
            # 只有当该行成功提取到至少一个单词时，才作为一个独立任务记录
            if words:
                current_parsed_tasks.append(str(words)) # 转换成 "['sky', 'blue']" 形式的字符串
        
        # 3. 如果提取到的任务清单发生了任何改变（新增行、增加了空格、或者回去修改了拼写）
        if current_parsed_tasks != last_parsed_tasks:
            last_parsed_tasks = current_parsed_tasks
            
            if current_parsed_tasks:
                print(f"\n[{time.strftime('%H:%M:%S')}] File change detected! Processing...")
                
                results = []
                for task in current_parsed_tasks:
                    # 如果是一个全新的单词组合，或是被修改后的新组合，调用模型
                    if task not in llm_cache:
                        t0 = time.time()
                        res = generate_correction(task)
                        t1 = time.time()
                        llm_cache[task] = res
                        print(f" -> [LLM Run] Processed {task} in {t1-t0:.3f}s")
                    
                    # 取出缓存的对应结果
                    results.append(llm_cache[task])
                
                # 4. 打印最终完整的输出，每行对应一个结果
                print("\n" + "="*40)
                print("           CURRENT OUTPUT           ")
                print("="*40)
                for r in results:
                    print(r)
                    print("-" * 20)

        # 轮询间隔
        time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        # 避免读取文件冲突时崩溃，静默等待下一次循环
        time.sleep(0.5)
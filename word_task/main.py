import time
from LLM import BiosensorLLM        # 这里将LLM的调用写成了一个类，调用的时候生成实例
from input import ensure_input_file, get_parsed_tasks
from output import write_to_output_file

INPUT_FILE = "./IO/input.txt"
OUTPUT_FILE = "./IO/output.txt"
MODEL_PATH = "../Models/LlaMa_8B"

def main():
    # 首先先确保input文件存在
    ensure_input_file(INPUT_FILE)
    
    # 创建实例，加载模型
    llm = BiosensorLLM(model_path=MODEL_PATH)
    
    print("\n" + "="*50)
    print("Biosensor Assistant Ready!")
    print(f"Monitoring '{INPUT_FILE}' -> Generating to '{OUTPUT_FILE}'")
    print("Press Ctrl+C to stop.")
    print("="*50)

    # 3. 记录状态与缓存
    last_parsed_tasks = []
    llm_cache = {}

    while True:
        try:
            # 调用input.py中的函数得到所有的要加工的task，
            # e.g.: ["['sky', 'blue', 'sun']", "['eat', 'apple']"]
            current_parsed_tasks = get_parsed_tasks(INPUT_FILE)
            
            if current_parsed_tasks is None:
                time.sleep(0.1)
                continue
            
            # 如果 input.txt 的提取内容发生变化
            if current_parsed_tasks != last_parsed_tasks:
                last_parsed_tasks = current_parsed_tasks
                
                if current_parsed_tasks:
                    print(f"\n[{time.strftime('%H:%M:%S')}] File change detected! Processing {len(current_parsed_tasks)} lines...")
                    
                    results = []
                    for task in current_parsed_tasks:
                        # 检查缓存，避免重复调用模型计算
                        if task not in llm_cache:
                            t0 = time.time()
                            res = llm.generate_correction(task)
                            t1 = time.time()
                            llm_cache[task] = res
                            print(f" -> [LLM Run] Processed {task} in {t1-t0:.3f}s")
                        
                        results.append(llm_cache[task])
                    
                    # 格式化并写入 output.txt
                    write_to_output_file(OUTPUT_FILE, results)
                    print(f" -> Successfully updated {OUTPUT_FILE}")

            # 轮询休眠
            time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Main loop error: {e}")
            time.sleep(0.5)

if __name__ == "__main__":
    main()
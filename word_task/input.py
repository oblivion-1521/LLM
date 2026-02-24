import os

def ensure_input_file(file_path):
    """create input file if not exists"""
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("")

def get_parsed_tasks(file_path):
    """
    读取文件并按行解析出任务列表，
    返回类似于 ["['sky', 'blue', 'sun']", "['eat', 'apple']"] 的列表
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 把content读出来之后分行
        lines = content.splitlines(keepends=True)
        current_parsed_tasks = []
        
        for line in lines:                  # 对所有行中的单独行，逐个提取words
            if line.endswith('\n'):
                # 已结束的句子
                words = line.strip().split()
            else:
                # 正在输入的行，遇到空格才提取
                last_space_idx = line.rfind(' ')
                if last_space_idx != -1:
                    words = line[:last_space_idx].strip().split()
                else:
                    words = []
            
            if words:
                current_parsed_tasks.append(str(words))
                
        return current_parsed_tasks
    except Exception as e:
        # 文件正被传感器占用写入时可能发生冲突
        return None
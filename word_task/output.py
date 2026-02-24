# 这个文件有两个函数，第一个用来去除LLM输出的废话
# 第二个用来统一输出格式
def parse_llm_result(llm_output):
    """
    将大模型的返回文本解析为单独的 单词列表 和 句子。
    输入示例: "Corrected words: ['sky', 'blue', 'sun']\nSentence: The sky is blue."
    返回: ("['sky', 'blue', 'sun']", "The sky is blue.")
    """
    corrected_words = ""
    sentence = ""
    
    lines = llm_output.split('\n')
    for line in lines:
        line_lower = line.lower()
        if line_lower.startswith("corrected words:"):
            # 截取冒号后面的内容
            corrected_words = line[len("corrected words:"):].strip()
        elif line_lower.startswith("sentence:"):
            sentence = line[len("sentence:"):].strip()
            
    return corrected_words, sentence

def write_to_output_file(file_path, llm_results):
    """
    将结果写入 output.txt。
    上半部分全是 Corrected words，下半部分全是 Sentence。
    """
    all_corrected_words = []
    all_sentences = []
    
    # 拆分结果
    for res in llm_results:
        cw, sent = parse_llm_result(res)
        if cw: all_corrected_words.append(cw)
        if sent: all_sentences.append(sent)
        
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Corrected words:\n")
            for cw in all_corrected_words:
                f.write(f"{cw}\n")
                
            f.write("\nSentence:\n")
            for sent in all_sentences:
                f.write(f"{sent}\n")
                
    except Exception as e:
        print(f"Error writing to output file: {e}")
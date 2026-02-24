from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

class BiosensorLLM:
    def __init__(self, model_path="./Models/LlaMa_8B"):
        # __init__ 负责初始化device、dtype、加载模型和tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        print("Loading model and tokenizer (safetensors + mmap)...")
        start_time = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=self.dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            device_map={"": self.device}
        )
        self.model.eval()
        print(f"Model loaded and moved to {self.device} in {time.time() - start_time:.2f} seconds.")

    def generate_correction(self, words_input: str) -> str:
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

        # 调用__init__时加载的tokenizer和model，generate prompt
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # inference step
        with torch.inference_mode():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False, 
                temperature=None,
                top_p=None,
                use_cache=True,
                eos_token_id=[
                    self.tokenizer.eos_token_id, 
                    self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ],
                pad_token_id=self.tokenizer.eos_token_id
            )

        # decode result
        result = self.tokenizer.decode(output_tokens[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        return result.strip()
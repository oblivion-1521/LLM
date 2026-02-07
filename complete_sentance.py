from google import genai
import random

# --- Configuration ---
# API_KEY = ""
FILE_PATH = "temp2.txt"

# Initialize Client
client = genai.Client(api_key=API_KEY)

def get_random_words(file_path, count):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        return random.sample(words, min(count, len(words)))
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []

def generate_sentence(words):
    word_str = ", ".join(words)
    # Instruction to the LLM
    prompt = f"Create a short, natural sentence using these words: {word_str}, don't add irrelebvant words, the sorter the better."
    
    try:
        # Use 'gemini-1.5-flash' - do NOT add 'models/' prefix here
        response = client.models.generate_content(
            model='gemini-2.0-flash-lite', 
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"API Error: {e}"

if __name__ == "__main__":
    # 1. Pick words
    # selected = get_random_words(FILE_PATH, random.randint(3, 5))
    selected = ["sky", "blue", "sun"]  # For testing, use fixed words
    if selected:
        print(f"Words: {selected}")
        
        # 2. Get sentence
        sentence = generate_sentence(selected)
        print(f"Result: {sentence}")
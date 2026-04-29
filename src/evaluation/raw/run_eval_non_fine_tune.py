import pandas as pd
import evaluate
from khmer_segmenter import Tokenizer

# --- CONFIGURATION ---
EVAL_FILES = [
    {"name": "Llama", "file": "outputs/non_fine_tune_prediction_llama.jsonl"},
    {"name": "Qwen", "file": "outputs/non_fine_tune_prediction_qwen.jsonl"},
    {"name": "Gemma", "file": "outputs/non_fine_tune_prediction_gemma.jsonl"}
]
REFERENCE_FILE = "../../data/raw/test_20percent_cleaned.jsonl"

def custom_tokenizer(text):
    return Tokenizer().tokenize(text)

def run_evaluations():
    # Load references once
    ref_df = pd.read_json(REFERENCE_FILE, lines=True)
    references = ref_df["title"].tolist()
    
    rouge_score = evaluate.load("rouge")
    all_results = {}

    for item in EVAL_FILES:
        try:
            pred_df = pd.read_json(item["file"], lines=True)
            predictions = pred_df["prediction"].tolist()

            results = rouge_score.compute(
                predictions=predictions,
                references=references,
                tokenizer=custom_tokenizer
            )
            
            all_results[item["name"]] = results
        except FileNotFoundError:
            print(f"Skipping {item['name']}: File not found.")

    # Print Comparison Table
    print(f"{'Model':<10} | {'ROUGE-1':<8} | {'ROUGE-2':<8} | {'ROUGE-L':<8}")
    print("-" * 45)
    for model_name, scores in all_results.items():
        print(f"{model_name:<10} | {scores['rouge1']*100:>7.1f} | {scores['rouge2']*100:>7.1f} | {scores['rougeL']*100:>7.1f}")

if __name__ == "__main__":
    run_evaluations()
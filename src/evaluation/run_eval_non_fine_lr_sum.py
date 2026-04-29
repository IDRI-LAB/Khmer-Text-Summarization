import pandas as pd
import evaluate
from khmer_segmenter import Tokenizer

# --- CONFIGURATION ---
EVAL_FILES = [
    {"name": "Llama", "file": "outputs/synthetic_non_fine_tune_prahok_llama.jsonl"},
    {"name": "Qwen", "file": "outputs/synthetic_non_fine_tune_prahok_qwen.jsonl"},
    {"name": "Gemma", "file": "outputs/synthetic_non_fine_tune_prahok_gemma.jsonl"}
]
REFERENCE_FILE = "../data/lr_sum/khm_test.jsonl"

# Khmer tokenizer
def custom_tokenizer(texts):
    tokenizer = Tokenizer()
    if isinstance(texts, str):
        return tokenizer.tokenize(texts)
    return [tokenizer.tokenize(t) for t in texts]

def run_evaluations():
    # Load references
    ref_df = pd.read_json(REFERENCE_FILE, lines=True)
    references = ref_df["summary"].tolist()
    n_references = len(references)
    
    rouge_score = evaluate.load("rouge")
    all_results = {}

    for item in EVAL_FILES:
        try:
            pred_df = pd.read_json(item["file"], lines=True)
            predictions = pred_df["prediction"].tolist()
            n_predictions = len(predictions)

            if n_predictions < n_references:
                print(f"Warning: {item['name']} predictions ({n_predictions}) < references ({n_references}).")
                print(f"ROUGE will be computed only for available predictions.")
                # truncate references to match predictions
                references_to_use = references[:n_predictions]
            elif n_predictions > n_references:
                print(f"Warning: {item['name']} predictions ({n_predictions}) > references ({n_references}).")
                print(f"Truncating predictions to match references.")
                predictions = predictions[:n_references]
                references_to_use = references
            else:
                references_to_use = references

            # Compute ROUGE
            results = rouge_score.compute(
                predictions=predictions,
                references=references_to_use,
                tokenizer=custom_tokenizer
            )

            all_results[item["name"]] = results
        except FileNotFoundError:
            print(f"Skipping {item['name']}: File not found.")

    # Print comparison table
    print(f"{'Model':<10} | {'ROUGE-1':<8} | {'ROUGE-2':<8} | {'ROUGE-L':<8}")
    print("-" * 45)
    for model_name, scores in all_results.items():
        print(f"{model_name:<10} | {scores['rouge1']*100:>7.2f} | {scores['rouge2']*100:>7.2f} | {scores['rougeL']*100:>7.2f}")

if __name__ == "__main__":
    run_evaluations()
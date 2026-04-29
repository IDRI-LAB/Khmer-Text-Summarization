from unsloth import FastLanguageModel
from datasets import Dataset
import pandas as pd
import json
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from src.utils import ALPACA_PROMPT

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# --- CONFIGURATION: list of models ---
MODELS_TO_RUN = [
    {
        "name": "llama",
        "model_name": "./outputs_Llama-3.2-3B-Instruct-bnb-4bit/checkpoint-1313",
        "output": "outputs/prediction_fine_tune_prahok_llama.jsonl"
    },
    {
        "name": "qwen",
        "model_name": "./outputs_Qwen2.5-7B-Instruct-bnb-4bit/checkpoint-3281",
        "output": "outputs/prediction_fine_tune_prahok_qwen.jsonl"
    },
    {
        "name": "gemma",
        "model_name": "./outputs_gemma-2b-bnb-4bit/checkpoint-1313",
        "output": "outputs/prediction_fine_tune_prahok_gemma.jsonl"
    }
]

DATASET_PATH = "PrahokBartDataset/khm_test.jsonl"


# --- Load dataset once ---
df_test = pd.read_json(DATASET_PATH, lines=True)
df_test = df_test.loc[:, ["summary", "text"]]
dataset = Dataset.from_pandas(df_test)

# --- Function to format prompts ---
def formatting_prompts_func(examples, alpaca_prompt):
    instructions = examples["text"]
    texts = []
    for ins in zip(instructions):
        text = alpaca_prompt.format(ins)
        texts.append(text)

    return {"text": texts}

# --- Apply formatting to dataset ---
fn_kwargs = {"alpaca_prompt": ALPACA_PROMPT}  # tokenizer not needed here
dataset = dataset.map(
    lambda x: formatting_prompts_func(x, **fn_kwargs),
    batched=True,
    remove_columns=["summary", "text"]
)

# --- Inference for multiple models ---
for model_cfg in MODELS_TO_RUN:
    print(f"--- Running model: {model_cfg['name']} ---")

    # Load model & tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["model_name"],
        max_seq_length=8192,
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)  # enable faster inference

    predictions = []
    with torch.no_grad():  # disable gradients

        for i in range(len(dataset)):
            # Tokenize input
            inputs_tokenized = tokenizer(
                dataset[i]["text"], return_tensors="pt", truncation=True, padding=True
            ).to("cuda")

            # Generate output
            outputs = model.generate(
                **inputs_tokenized,
                max_new_tokens=128,
                use_cache=True,
                do_sample=True,
                temperature=0.3,
                top_p=0.85
            )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            clean = decoded.split("### Response:")[-1].strip()
            predictions.append(clean)

            if (i + 1) % 50 == 0:
                print(f"{i+1}/{len(dataset)} examples processed...")

    # Save predictions for this model
    with open(model_cfg["output"], "w", encoding="utf-8") as f:
        for text in predictions:
            json.dump({"prediction": text}, f, ensure_ascii=False)
            f.write("\n")
            
    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()

print("✅ All models finished.")
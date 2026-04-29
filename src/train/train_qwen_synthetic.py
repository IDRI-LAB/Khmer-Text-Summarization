# ============================================================
# Khmer Text Summarization - Qwen 2.5 7B (Synthetic Dataset)
# Fine-tuning with QLoRA using Unsloth
# Inspired by Unsloth's fine-tuning notebooks:
# https://github.com/unslothai/unsloth
# ============================================================

import os
import torch
import pandas as pd

from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer,SFTConfig
from unsloth import UnslothTrainingArguments
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.utils import ALPACA_PROMPT
# -------------------------------------------------------------------------
# Environment (IMPORTANT)
# -------------------------------------------------------------------------
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# Model
MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 512            # 🔥 DO NOT use 2048 on 16GB
DTYPE = None
LOAD_IN_4BIT = True

# LoRA
LORA_R = 8                      # safer
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Training
# This setting optimize for 16GB VRAM, Model Gemma-2b   
OUTPUT_DIR = "outputs" + '_' + MODEL_NAME.split("/")[-1]
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4 # Setting adjust to Gemma-2B, vram 16gb
# RANDOM_NUM_PICK_EVAL = 6400 # Random pick up the number of eval dataset, can't take full eval dataset because of GPU server's speed issue
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 1
MAX_STEPS = -1 #-1 change to -1 for full training dataset
LOGGING_STEPS = 25
EVAL_STEPS = 500
SAVE_STEPS = 500

# -------------------------------------------------------------------------
# Prompt Formatting
# -------------------------------------------------------------------------
def formatting_prompts_func(examples, tokenizer, alpaca_prompt):
    eos = tokenizer.eos_token
    instructions = examples["content"]
    # inputs       = examples["input"]
    outputs = examples["summary"]

    texts = []
    for ins, out in zip(instructions, outputs):
        text = alpaca_prompt.format(ins, out) + eos
        labels = out
        texts.append(text)

    return {"text": texts}

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")

    # 1️⃣ Load model & tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    # 2️⃣ Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    train_df = pd.read_json("synthetic_dataset/7000_train_cleaned.jsonl", lines=True)
    val_df = pd.read_json("synthetic_dataset/1000_val_cleaned.jsonl", lines=True)
    train_df = train_df.loc[:, ["summary", "content"]]
    train_df = train_df.dropna(subset=["summary", "content"])
    dataset = Dataset.from_pandas(train_df)


    val_df = val_df.loc[:, ["summary", "content"]]
    val_df = val_df.dropna(subset=["summary", "content"])
    val_dataset = Dataset.from_pandas(val_df)



    fn_kwargs = {"tokenizer": tokenizer, "alpaca_prompt": ALPACA_PROMPT}
    dataset = dataset.map(lambda x: formatting_prompts_func(x, **fn_kwargs),
        batched=True,
        remove_columns=["summary", "content"],
    )

    val_dataset = val_dataset.map(lambda x: formatting_prompts_func(x, **fn_kwargs),
        batched=True,
        remove_columns=["summary", "content"],
    )

    # 4️⃣ Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        # eval_dataset=val_dataset.select(range(RANDOM_NUM_PICK_EVAL)),
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        # dataset_num_proc=1, # Enable this if you have enought CPU's core, disable if you are an only user that is using this server (to speed up)
        packing=True,                     # 🔥 VERY IMPORTANT
        args=UnslothTrainingArguments(
            per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=10,
            num_train_epochs=NUM_TRAIN_EPOCHS,
            max_steps=MAX_STEPS,
            learning_rate=LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=LOGGING_STEPS,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            group_by_length=True,         # 🔥 prevents padding OOM
            save_steps=SAVE_STEPS,
            report_to="tensorboard",
            logging_dir=OUTPUT_DIR,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            eval_accumulation_steps = 1,   # PREVENT MEMORY CRASH
            per_device_eval_batch_size = PER_DEVICE_EVAL_BATCH_SIZE,
            save_total_limit=2,
            # dataset_num_proc=1,           # 🔥 force single-process tokenization
    ),
    )

    # 5️⃣ Train
    print("🔥 Training started...")
    trainer.train()
    print("✅ Training complete!")

    # 6️⃣ Save LoRA
    print("💾 Saving LoRA adapters...")
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")

    print("🎉 Done!")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()

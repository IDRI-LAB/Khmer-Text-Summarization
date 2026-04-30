# 🇰🇭 Khmer Text Summarization

Fine-tuning large language models for **Khmer text summarization** using QLoRA (4-bit) with [Unsloth](https://github.com/unslothai/unsloth).

## 📌 Overview

This project fine-tunes 3 models on 2 different Khmer datasets and evaluates them using ROUGE scores.

| Model | Base Model |
|-------|-----------|
| Gemma | unsloth/gemma-2b-bnb-4bit |
| LLaMA | unsloth/Llama-3.2-3B-Instruct-bnb-4bit |
| Qwen | unsloth/Qwen2.5-7B-Instruct-bnb-4bit |


---

## 🛢️Dataset
    The dataset is available but requires to request. Please contact chily.ran@student.cadt.edu.kh for access.
| Dataset | Description |
|---------|-------------|
| Raw (Title-based) | 144k Real Khmer news articles |
| Synthetic | 10k synthetic Khmer samples generate from Gemini 2.5 Flash-Lite |

## 📂 Project Structure

```

khmer-summarization-llm/
├── data/                          # Dataset (ignored in Git)
├── src/
│   ├── train/                     # Training scripts
│   │   ├── train_gemma_raw.py
│   │   ├── train_gemma_synthetic.py
│   │   ├── train_llama_raw.py
│   │   ├── train_llama_synthetic.py
│   │   ├── train_qwen_raw.py
│   │   └── train_qwen_synthetic.py
│   ├── inference/                 # Inference scripts
│   │   ├── run_eval_non_fine_lr_sum.py
│   │   ├── raw/
│   │   │   ├── run_inference_test_set.py
│   │   │   ├── run_inference_non_fine_tune.py
│   │   │   └── run_inference_lr_sum.py
│   │   └── synthetic/
│   │       ├── run_inference_test_set.py
│   │       ├── run_inference_non_fine_tune.py
│   │       └── run_inference_lr_sum.py
│   ├── evaluation/                # Evaluation scripts
│   │   ├── run_inference_non_fine_lr_sum.py
│   │   ├── raw/
│   │   │   ├── run_eval_fine_tune.py
│   │   │   ├── run_eval_non_fine_tune.py
│   │   │   └── run_eval_lr_sum.py
│   │   └── synthetic/
│   │       ├── run_eval_fine_tune.py
│   │       ├── run_eval_non_fine_tune.py
│   │       └── run_eval_lr_sum.py
│   └── utils.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/ChilyRan/khmer-text-summarization.git
cd khmer-text-summarization
pip install -r requirements.txt
```

---

## 🚀 Training

### Raw Dataset
```bash
python src/train/train_gemma_raw.py
python src/train/train_llama_raw.py
python src/train/train_qwen_raw.py
```

### Synthetic Dataset
```bash
python src/train/train_gemma_synthetic.py
python src/train/train_llama_synthetic.py
python src/train/train_qwen_synthetic.py
```

---

## 🔍 Inference

### Fine-tuned Models
```bash
# Raw
python src/inference/raw/run_inference_test_set.py
python src/inference/raw/run_inference_lr_sum.py

# Synthetic
python src/inference/synthetic/run_inference_test_set.py
python src/inference/synthetic/run_inference_lr_sum.py

```

### Non Fine-tuned (Baseline)
```bash
# Raw
python src/inference/raw/run_inference_non_fine_tune.py

# Synthetic
python src/inference/synthetic/run_inference_non_fine_tune.py

# LR-Sum
python src/inference/run_inference_non_fine_lr_sum.py

```

---

## 📊 Evaluation

### Fine-tuned Models
```bash
# Raw
python src/evaluation/raw/run_eval_fine_tune.py
python src/evaluation/raw/run_eval_lr_sum.py

# Synthetic
python src/evaluation/synthetic/run_eval_fine_tune.py
python src/evaluation/synthetic/run_eval_lr_sum.py
```

### Non Fine-tuned (Baseline)
```bash
# Raw
python src/evaluation/raw/run_eval_non_fine_tune.py

# Synthetic
python src/evaluation/synthetic/run_eval_non_fine_tune.py

# LR-Sum
python src/inference/run_eval_non_fine_lr_sum.py

```

---

## 🤗 HuggingFace Adapters

Trained adapters are available on HuggingFace:

| Model | Repo |
|-------|------|
| Gemma | [ChilyRan/gemma-khmer-adapters](https://huggingface.co/ChilyRan/gemma-khmer-adapters) |
| LLaMA | [ChilyRan/llama-khmer-adapters](https://huggingface.co/ChilyRan/llama-khmer-adapters) |
| Qwen | [ChilyRan/qwen-khmer-adapters](https://huggingface.co/ChilyRan/qwen-khmer-adapters) |

### Load and run inference:

```python
from unsloth import FastLanguageModel
import torch

ALPACA_PROMPT = """ខាងក្រោមនេះគឺជាសេចក្តីណែនាំអំពីកិច្ចការមួយ។ សូមផ្តល់ចម្លើយឱ្យបានត្រឹមត្រូវ ពេញលេញ និងងាយយល់។  

### Instruction:
ចូលសង្ខេប អត្ថបទខាងក្រោមនេះ
### Input:
{}
### Response:
"""

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2b-bnb-4bit",
    max_seq_length=8192,
    load_in_4bit=True,
    adapter_name="ChilyRan/gemma-khmer-adapters",
    adapter_kwargs={"subfolder": "synthetic"}  # or "title_based"
)
FastLanguageModel.for_inference(model)

text = "បញ្ចូលអត្ថបទខ្មែររបស់អ្នកនៅទីនេះ..."
prompt = ALPACA_PROMPT.format(text)
inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True,
        do_sample=True,
        temperature=0.3,
        top_p=0.85
    )

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
summary = decoded.split("### Response:")[-1].strip()
print(summary)
```

---

## 📋 Prompt Format
"""ខាងក្រោមនេះគឺជាសេចក្តីណែនាំអំពីកិច្ចការមួយ។ សូមផ្តល់ចម្លើយឱ្យបានត្រឹមត្រូវ ពេញលេញ និងងាយយល់។  

### Instruction:
ចូលសង្ខេប អត្ថបទខាងក្រោមនេះ
### Input:
{}
### Response:
"""

---

## 🔧 Training Details

| Config | Value |
|--------|-------|
| Method | QLoRA |
| Framework | Unsloth |
| Quantization | 4-bit |
| Task | Khmer text summarization |

---
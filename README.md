# tune_quest_hackathon
Hackathon on fine tuning of local weak LLM to  predict semantic correctness of two given code 


# 🚀 Cross-Language Code Equivalence Detection using Gemma-3

**AI Club Hackathon Project**

This repository contains the code and fine-tuned model for detecting semantic equivalence between code snippets. Built using Google's **Gemma-3-1b-it**, the model evaluates pairs of source code (Python-Python, Java-Java, and Python-Java) and predicts whether they implement the same underlying logic (`Equivalent` or `Not Equivalent`).

## 🌟 Key Highlights
* **Resource-Constrained Training:** Successfully fine-tuned a 1-Billion parameter LLM on a consumer-grade GPU (NVIDIA RTX 3050 with 6GB VRAM) without encountering OOM crashes.
* **Incremental Training Pipeline:** Utilized a multi-stage, incremental learning approach to continuously improve the model's F1-score across large datasets.
* **Leakage-Free Validation:** Implemented strict data stratification and leakage-prevention split strategies to ensure robust and authentic validation metrics.

## 🛠️ Tech Stack & Optimizations
* **Model:** `google/gemma-3-1b-it`
* **Frameworks:** PyTorch, Hugging Face `transformers`, `peft`, `datasets`
* **Quantization:** 4-bit NormalFloat (NF4) via `bitsandbytes`
* **Fine-Tuning Method:** QLoRA (Quantized Low-Rank Adaptation)
* **Memory Optimizations:** 
  * `gradient_checkpointing=True`
  * `optim="paged_adamw_8bit"` (to handle memory spikes)
  * Left-padding for optimized batch generation.

## 📊 Dataset & Preprocessing
The dataset consists of code pairs across multiple languages. Key preprocessing steps included:
1. **Deduplication & Stratification:** Ensured that functions present in the training set did not leak into the validation set.
2. **Prompt Engineering:** Structured the inputs into strict system/user chat templates optimal for Gemma-3's instruction-tuned architecture.
3. **Target Formatting:** Model was trained to generate concise classification tokens (e.g., yielding "equivalent" or "not equivalent").

## 📈 Results
The model was evaluated using the **F1-Score** to handle any class imbalances in the equivalence dataset. 
* **Baseline F1-Score:** ~0.8964
* **Performance Breakdown:** Maintained consistent accuracy across identical (`python-python`, `java-java`) and cross-language (`java-python`) pairs. *(Update your final round 2 F1 score here!)*

## 🚀 How to Run (Inference)

### 1. Install Dependencies
```bash
pip install torch transformers peft bitsandbytes accelerate scikit-learn

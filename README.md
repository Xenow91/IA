# AI Project: 450M Parameter Language Model from Scratch

This project contains the complete source code to create, pre-train, and fine-tune (SFT) a GPT-style Large Language Model (LLM) from scratch using PyTorch. To try it out: https://xenow91.github.io/vitrine/index.html

## Project Objective
The objective is to demonstrate that it is possible to train a 450 Million parameter model on 12 Billion tokens using consumer-grade hardware (a single RTX 5060 Ti 16GB VRAM) via the Vast.ai platform, applying state-of-the-art techniques (FlashAttention, bfloat16, RMSNorm, RoPE). Above all, it is also a personal endeavor to improve my skills and gain a deeper understanding of this field that I am passionate about!

## Model Characteristics
- Architecture: Transformer
- Parameters: ~446 Million
- Layers: 26
- Attention Heads: 16
- Embedding Dimension (d_model): 1024
- Vocabulary Size: 50,304 tokens
- Maximum Context: 2048 tokens

## Implemented Technical Optimizations
The code integrates modern optimizations to maximize speed and reduce memory footprint:
1. FlashAttention (SDP): Hardware acceleration for attention computation.
2. Mixed Precision (bfloat16): Halves VRAM consumption without losing stability.
3. Gradient Accumulation: Simulates a global batch of 256 sequences to stabilize gradient descent despite the 16GB VRAM limit.
4. Fused AdamW & Cosine Decay: C++ fused optimizer and dynamic Learning Rate decay to converge to the best Loss.
5. Rotary Positional Embeddings (RoPE) and RMSNorm for better mathematical stability.

## Project Structure

### 1. Tokenizer and Data Preparation
For educational purposes, a custom tokenizer (`custom_tokenizer`) was developed to understand the underlying mechanics of Byte-Pair Encoding (BPE). However, for performance and standardization reasons, the final project uses OpenAI's optimized tokenizer (`tiktoken` GPT-2 version).
The `prepare.py` script handles downloading and tokenizing a subset of the HuggingFaceFW/fineweb-edu dataset (high-quality educational texts), generating binary files `train.bin` and `val.bin` for ultra-fast reading via `np.memmap`.

### 2. Model Architecture (model.py)
Complete definition of the neural network with PyTorch. No hidden third-party libraries; the entire architecture (Attention, FeedForward, Normalization) is written from scratch.

### 3. Pre-Training (train.py)
Intensive training script designed to run on Vast.ai. Includes live tracking on WandB and an automatic backup system for the best checkpoint (`best_model.pt`) every time a Loss record is broken.

### 4. Instruction Tuning / SFT (prepare_sft.py & finetune.py)
Once the model has learned to generate text (Pre-training), these scripts transform it into an Assistant.
- Downloads the Databricks Dolly-15k dataset.
- Applies Loss Masking (`ignore_index=-1`) on the user's instructions to force the model to learn solely from the responses.
- Slow learning (Low Learning Rate) over 1.5 Epochs to avoid Catastrophic Forgetting.

## Training Reproduction
1. Install dependencies (`torch`, `numpy`, `datasets`, `tiktoken`, `wandb`).
2. Run `python prepare.py` to download the 20GB of pre-training data.
3. Transfer the folder to your GPU machine (e.g., Vast.ai) via SFTP/WinSCP.
4. Run `python train.py`. The training takes approximately 9 days on an RTX 5060 Ti.
5. Upon completion, run `python prepare_sft.py` then `python finetune.py` for the instruction-tuning phase.

---
Project created for educational and research purposes on LLMs.

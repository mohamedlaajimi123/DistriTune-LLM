# DistriTune-LLM

A clean, end-to-end engineering pipeline built to train, evaluate, and optimize open-source Large Language Models (LLMs). 

### What It Does:
1. **Speeds Up Training:** Uses PyTorch DDP to split data across multiple GPUs, making model fine-tuning much faster.
2. **Automates Evaluation:** Plugs into a RAG gateway to automatically test the model's accuracy and block hallucinations.
3. **Optimizes Prompts:** Uses an automated loop to test and rewrite system prompts, replacing manual guesswork with code.
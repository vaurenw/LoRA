# LoRA Fine-tuning Pipeline for Llama 3.2-1B inspired by @nicknochnack

## Components

### datagen.py

- PDF document processing with Docling DocumentConverter
- Automated Q&A pair generation using Qwen2.5-Coder 14B
- Structured JSON output for training

### train.py

- Fine-tunes Llama 3.2-1B using Parameter-Efficient Fine-Tuning (PEFT)
- Implements 4-bit quantization for memory efficiency
- Uses chat formatting for instruction-following behavior

**Key Configuration:**
- **Base Model:** meta-llama/Llama-3.2-1B
- **LoRA Parameters:**
  - Rank (r): 32
  - Alpha: 64
  - Dropout: 0.05
  - Target modules: all-linear layers
- **Training:**
  - 4-bit quantization (NF4)
  - Gradient checkpointing
  - 3 epochs with batch size 2

**Memory Optimization:**
- BitsAndBytesConfig for 4-bit quantization
- Gradient checkpointing for reduced memory usage
- Automatic garbage collection and cache clearing

### Training Parameters

- **Quantization:** 4-bit NF4 with double quantization
- **Batch Size:** 2 (adjustable based on GPU memory)
- **Epochs:** 3
- **Learning Rate:** Default SFT configuration
- **Mixed Precision:** FP16



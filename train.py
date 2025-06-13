from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch
import os

# Load dataset
dataset = load_dataset("data", split="train")

# Chat formatting
def format_chat_template(batch, tokenizer):
    system_prompt = """You are a helpful, honest and harmless assistant designed to help engineers. Think through each question logically and provide an answer. Don't make things up; if you're unable to answer a question, say so clearly."""

    samples = []
    questions = batch["question"]
    answers = batch["answer"]

    for q, a in zip(questions, answers):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
            "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
            "{{ content }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        )
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        samples.append(text)

    return {
        "instruction": questions,
        "response": answers,
        "text": samples,
    }


base_model = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(base_model, token="", trust_remote_code=True)


train_dataset = dataset.map(lambda x: format_chat_template(x, tokenizer), batched=True)


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto",
    token="",
    trust_remote_code=True,
    cache_dir="./hf_cache", 
)

# LoRA prep
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir="llama3-1B-sft",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_total_limit=1,
    save_strategy="epoch",
    report_to=[],
    remove_unused_columns=False,
    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    peft_config=peft_config,
)


trainer.train()
trainer.save_model("llama3-1B-final")


import gc
gc.collect()
torch.cuda.empty_cache()

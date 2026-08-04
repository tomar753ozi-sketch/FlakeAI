"""FlakeAI Quick Test - 2 dakika"""
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline
)

# Veri
print("Veri yukleniyor...")
texts = []
with open("data/training.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line and len(line) > 10:
            texts.append(line)
texts = texts[:5000]  # Hizli test icin 5K
print(f"  {len(texts)} satir")

dataset = Dataset.from_dict({"text": texts})

# Model
print("\nModel yukleniyor...")
model_name = "ytu-ce-cosmos/turkish-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
print(f"  {model.num_parameters()/1e6:.1f}M parametre")

# Tokenize
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Egitim - 2 dakika sinirli
training_args = TrainingArguments(
    output_dir="./checkpoints-quick",
    max_steps=100,  # 100 step = ~2 dk
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    logging_steps=10,
    fp16=True,
    save_strategy="no",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Egit
print("\nEgitim basliyor (2 dk sinirli)...")
trainer.train()

# Kaydet
model.save_pretrained("./flakeai-quick")
tokenizer.save_pretrained("./flakeai-quick")
print("\nKaydedildi: ./flakeai-quick")

# Test
print("\n" + "="*50)
print("TEST")
print("="*50)

chat = pipeline("text-generation", model="./flakeai-quick", tokenizer=tokenizer, device=0)

tests = [
    "def fibonacci(",
    "class Stack:",
    "Merhaba nasilsin?",
    "Python ile program",
    "Yapay zeka",
    "def quicksort(",
]

for prompt in tests:
    result = chat(prompt, max_length=100, do_sample=True, temperature=0.7, top_p=0.9)
    print(f"\n>>> {prompt}")
    print(f"    {result[0]['generated_text'][:200]}")

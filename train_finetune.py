"""FlakeAI Fine-tune Script - Turkish GPT-2 Base

ytu-ce-cosmos/turkish-gpt2 modelini senin verinle inceltiyor.
Sonuç: Türkçe konuşan, kod yazabilen chatbot.
"""
import os
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


def load_data(data_path="data/training.txt", max_samples=60000):
    print("Veri yukleniyor...")
    texts = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and len(line) > 10:
                texts.append(line)
    texts = texts[:max_samples]
    print(f"  {len(texts)} satir yuklendi")
    return texts


def train():
    print("=" * 60)
    print("FlakeAI Fine-tune (Turkish GPT-2 Base)")
    print("=" * 60)

    # Veri
    texts = load_data()
    dataset = Dataset.from_dict({"text": texts})

    # Model & Tokenizer - Turkish GPT-2
    print("\nModel yukleniyor... (ytu-ce-cosmos/turkish-gpt2)")
    model_name = "ytu-ce-cosmos/turkish-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    print(f"  Parametre: {model.num_parameters():,} ({model.num_parameters()/1e6:.1f}M)")

    # Tokenize
    print("\nTokenize ediliyor...")
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length"
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training args
    print("\nEgitim ayarlari...")
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        save_strategy="epoch",
        logging_steps=50,
        fp16=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Egit
    print("\n" + "=" * 60)
    print("Fine-tune basliyor! (~1.5-2 saat)")
    print("=" * 60)
    trainer.train()

    # Kaydet
    print("\nModel kaydediliyor...")
    final_dir = "./flakeai-final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"  Kaydedildi: {final_dir}")

    # Test
    print("\n" + "=" * 60)
    print("Test")
    print("=" * 60)
    from transformers import pipeline
    chat = pipeline("text-generation", model=final_dir, tokenizer=tokenizer, device=0)

    test_prompts = [
        "Merhaba, nasilsin?",
        "Python ile program yazmak istiyorum.",
        "Yapay zeka nedir?",
        "def fibonacci(",
        "Bugun hava cok guzel.",
        "FlakeAI kimdir?",
    ]

    for prompt in test_prompts:
        result = chat(
            prompt,
            max_length=150,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        print(f"\nSoru: {prompt}")
        print(f"Cevap: {result[0]['generated_text'][:300]}")

    # Boyut
    total_size = sum(
        os.path.getsize(os.path.join(final_dir, f))
        for f in os.listdir(final_dir)
        if os.path.isfile(os.path.join(final_dir, f))
    )
    print(f"\n{'=' * 60}")
    print(f"Model boyutu: {total_size/1e6:.1f} MB")
    print(f"Kayit: {final_dir}")
    print("=" * 60)


if __name__ == "__main__":
    train()

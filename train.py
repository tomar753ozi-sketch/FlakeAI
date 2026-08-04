"""FlakeAI 200M Model Training Script

Kullanım:
    python train.py --data data/training.txt --epochs 5

Önce veriyi hazırla:
    python prepare_data.py
"""
import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flakeai.model.transformer import FlakeAIModel, ModelConfig
from flakeai.model.tokenizer import FlakeAITokenizer


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=256):
        self.encodings = []
        skipped = 0
        for text in texts:
            if not text or len(text.strip()) < 5:
                skipped += 1
                continue
            ids = tokenizer.encode(text)
            if len(ids) > 5:
                self.encodings.append(ids[:max_len])
            else:
                skipped += 1
        print(f"  Dataset: {len(self.encodings)} samples, {skipped} skipped")

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        ids = self.encodings[idx]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def collate_fn(batch):
    max_len = max(len(x) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad_len = max_len - len(x)
        xs.append(F.pad(x, (0, pad_len), value=0))
        ys.append(F.pad(y, (0, pad_len), value=-100))
    return torch.stack(xs), torch.stack(ys)


def load_data(data_path: str) -> list:
    texts = []
    print(f"Veri yukleniyor: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and len(line) > 3:
                texts.append(line)
    print(f"  {len(texts)} satir yuklendi")
    return texts


def train(
    data_path: str = "data/training.txt",
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 3e-4,
    save_dir: str = "checkpoints",
    resume: str = None,
):
    print("=" * 60)
    print("FlakeAI 200M Model Training")
    print("=" * 60)

    config = ModelConfig()

    print(f"\nModel Config:")
    print(f"  vocab_size: {config.vocab_size:,}")
    print(f"  d_model: {config.d_model}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  n_kv_heads: {config.n_kv_heads}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  dropout: {config.dropout}")
    print(f"  Total params: {config.count_parameters():,} ({config.count_parameters()/1e6:.1f}M)")

    tokenizer = FlakeAITokenizer(config.vocab_size)
    texts = load_data(data_path)
    tokenizer.train(texts)
    print(f"  Vocab size: {len(tokenizer)}")

    model = FlakeAIModel(config)
    actual_params = model.get_num_params()
    print(f"  Actual params: {actual_params:,} ({actual_params/1e6:.1f}M)")

    start_epoch = 0
    start_batch = 0
    best_loss = float('inf')
    ckpt = None

    if resume and os.path.exists(resume):
        print(f"\nCheckpoint yukleniyor: {resume}")
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"]
        if "batch" in ckpt:
            start_batch = ckpt["batch"]
        if "best_loss" in ckpt:
            best_loss = ckpt["best_loss"]
        print(f"  Epoch {start_batch}'den devam ediliyor")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    model.to(device)

    print("\nDataset olusturuluyor...")
    dataset = TextDataset(texts, tokenizer, config.max_seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
    )

    print("\nOptimizer hazirlaniyor...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    if ckpt is not None:
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        ckpt = None

    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Training basliyor! {epochs} epoch, batch_size={batch_size}")
    print(f"  Batch sayisi/epoch: {len(dataset) // batch_size}")
    print(f"{'=' * 60}\n")

    training_start = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        epoch_start = time.time()

        for batch_idx, (x, y) in enumerate(dataloader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            x, y = x.to(device), y.to(device)

            _, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                lr_now = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f} | LR: {lr_now:.6f}")

                # Her 100 batch'te checkpoint kaydet
                torch.save({
                    "epoch": epoch,
                    "batch": batch_idx + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss.item(),
                    "best_loss": best_loss,
                    "config": config.__dict__,
                }, os.path.join(save_dir, "flakeai_last.pt"))

        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - training_start

        print(f"\nEpoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s | Total: {total_time/60:.1f}min")

        # Epoch checkpoint
        torch.save({
            "epoch": epoch + 1,
            "batch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "best_loss": best_loss,
            "config": config.__dict__,
        }, os.path.join(save_dir, "flakeai_last.pt"))

        # Epoch isimli checkpoint
        named_path = os.path.join(save_dir, f"flakeai_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "loss": avg_loss,
            "config": config.__dict__,
        }, named_path)
        print(f"  Checkpoint: {named_path}")

        # Best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "flakeai_best.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss,
                "config": config.__dict__,
            }, best_path)
            print(f"  En iyi model guncellendi: loss={avg_loss:.4f}")

    # Final
    print("\n" + "=" * 60)
    print("Training tamamlandi!")
    print("=" * 60)

    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)

    final_path = os.path.join(save_dir, "flakeai_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
    }, final_path)

    total_time = time.time() - training_start
    print(f"\nOzet:")
    print(f"  Toplam sure: {total_time/60:.1f} dakika")
    print(f"  Epoch sayisi: {epochs}")
    print(f"  Son loss: {avg_loss:.4f}")
    print(f"  En iyi loss: {best_loss:.4f}")
    print(f"  Model: {final_path}")

    # Test
    print("\n" + "=" * 60)
    print("Test")
    print("=" * 60)
    model.eval()
    test_prompts = [
        "Merhaba, nasilsin?",
        "Python ile program yazmak",
        "Yapay zeka",
        "def fibonacci(",
        "Bugun hava",
    ]
    for prompt in test_prompts:
        input_ids = tokenizer.encode(prompt)
        x = torch.tensor([input_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            output = model.generate(x, max_new_tokens=50, temperature=0.8)
        generated = tokenizer.decode(output[0].tolist())
        print(f"\nPrompt: {prompt}")
        print(f"Output: {generated[:200]}")

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FlakeAI model")
    parser.add_argument("--data", type=str, default="data/training.txt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        resume=args.resume,
    )

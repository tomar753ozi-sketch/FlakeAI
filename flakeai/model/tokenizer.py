"""FlakeAI Tokenizer - Basit BPE Tokenizer"""
import json
from pathlib import Path


class FlakeAITokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.inverse_vocab = {}

    def train(self, texts):
        """Basit kelime tabanlı tokenizer eğitimi"""
        word_freqs = {}
        for text in texts:
            for word in text.split():
                word = " ".join(list(word)) + " </w>"
                word_freqs[word] = word_freqs.get(word, 0) + 1

        # Tüm karakterleri topla
        chars = set()
        for word in word_freqs:
            for c in word.split():
                if c != "</w>":
                    chars.add(c)

        # Vocab oluştur
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        idx = 4
        for c in sorted(chars):
            if c not in self.vocab:
                self.vocab[c] = idx
                idx += 1

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = {}

    def encode(self, text):
        """Metni token'lara çevir"""
        tokens = [self.vocab.get("<bos>", 1)]
        for char in text:
            tokens.append(self.vocab.get(char, self.vocab["<unk>"]))
        tokens.append(self.vocab.get("<eos>", 2))
        return tokens

    def decode(self, ids):
        """Token'ları metne çevir"""
        tokens = []
        for id in ids:
            if id in self.inverse_vocab:
                token = self.inverse_vocab[id]
                if token not in ("<pad>", "<bos>", "<eos>"):
                    tokens.append(token)
        return "".join(tokens).replace(" </w>", " ").strip()

    def save(self, path):
        """Tokenizer'ı kaydet"""
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f)

    def load(self, path):
        """Tokenizer'ı yükle"""
        with open(path, "r") as f:
            data = json.load(f)
            self.vocab = data.get("vocab", {})
            self.merges = data.get("merges", {})
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return len(self.vocab)

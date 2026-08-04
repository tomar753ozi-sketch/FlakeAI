"""200M Parameter Transformer Model for FlakeAI.

LLaMA/Mistral/Phi'den ogrenilen modern teknikler:
- RMSNorm (pre-normalization)
- SwiGLU activation
- Rotary Positional Embeddings (RoPE)
- Grouped Query Attention (GQA)
- Bias-free linear layers
- Weight tying
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 64000
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    n_layers: int = 12
    d_ff: int = 2048
    max_seq_len: int = 256
    dropout: float = 0.1
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def count_parameters(self) -> int:
        embed = self.vocab_size * self.d_model + self.max_seq_len * self.d_model
        attn_q = self.d_model * self.d_model * self.n_layers
        attn_kv = 2 * self.d_model * (self.head_dim * self.n_kv_heads) * self.n_layers
        attn_o = self.d_model * self.d_model * self.n_layers
        swiglu = 3 * self.d_model * self.d_ff * self.n_layers
        rms = 2 * self.d_model * self.n_layers + self.d_model
        return embed + attn_q + attn_kv + attn_o + swiglu + rms


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq_.size(2)]
    return torch.view_as_real(xq_ * freqs_cis).flatten(-2).type_as(xq), torch.view_as_real(xk_ * freqs_cis).flatten(-2).type_as(xk)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask=None, freqs_cis=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        att = (q @ k.transpose(-2, -1)) / self.scale
        if mask is not None:
            att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = int(2 / 3 * config.d_ff * 2)
        hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config)
        self.ln2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)

    def forward(self, x, mask=None, freqs_cis=None):
        x = x + self.attn(self.ln1(x), mask, freqs_cis)
        x = x + self.ffn(self.ln2(x))
        return x


class FlakeAIModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_final = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(config.head_dim, config.max_seq_len * 2, config.rope_theta),
            persistent=False,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embed(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        pos_emb = self.pos_embed(pos)
        x = self.embed_dropout(tok_emb + pos_emb)

        mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
        freqs_cis = self.freqs_cis[:T]

        for block in self.blocks:
            x = block(x, mask, freqs_cis)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        self.train()
        return idx

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_config(self) -> dict:
        return {
            "vocab_size": self.config.vocab_size,
            "d_model": self.config.d_model,
            "n_heads": self.config.n_heads,
            "n_kv_heads": self.config.n_kv_heads,
            "n_layers": self.config.n_layers,
            "d_ff": self.config.d_ff,
            "max_seq_len": self.config.max_seq_len,
            "dropout": self.config.dropout,
            "num_params": self.get_num_params(),
        }


def quantize_model(model: FlakeAIModel, bits: int = 8) -> FlakeAIModel:
    if bits == 8:
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
    elif bits == 4:
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.float16
        )
    return model


def save_quantized(model: FlakeAIModel, path: str, bits: int = 8):
    quantized = quantize_model(model, bits)
    torch.save({"model_state_dict": quantized.state_dict(), "quantized": True, "bits": bits}, path)
    return path

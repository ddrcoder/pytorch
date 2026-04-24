"""xsa_head_diversity_experiment.py

Mini GPT experiment that evaluates a cross-head attention-overlap
regularizer. The regularizer penalises the average pairwise dot product of
causal attention probabilities across heads, with a per-query baseline so
uniform causal attention is 0.

For each query position i in an attention map A[B, H, T, T]:
    V = A[b, :, i, :]                         # [H, T], row-stochastic on <=i+1 keys
    cross = (V.sum(0)**2).sum() - (V**2).sum() # = sum_{h != h'} <V_h, V_h'>
    avg   = cross / (H * (H - 1))
    corrected = avg - 1 / (i + 1)              # subtract uniform baseline
Loss is corrected averaged over batch, query positions, and layers.

Research questions the sweep is meant to illuminate:
  * Does the regularizer actually reduce cross-head overlap?
  * Does it raise attention entropy or only diversify peaks?
  * Does validation LM loss improve, stay flat, or degrade?
  * Which lambda is the best tradeoff?

Run:
    python xsa_head_diversity_experiment.py --sweep
    python xsa_head_diversity_experiment.py --lambda-div 1e-3
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# -----------------------------------------------------------------------------
# Mini GPT
# -----------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=True)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=True)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)
        self.last_attn: Optional[torch.Tensor] = None  # cached [B, H, T, T]

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        # [B, H, T, Dh]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, T, T]
        att = att.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        self.last_attn = att if store_attn else None
        y = att @ v  # [B, H, T, Dh]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, store_attn: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), store_attn=store_attn)
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied LM head
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        store_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[torch.Tensor]]:
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, store_attn=store_attn)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        attns = [blk.attn.last_attn for blk in self.blocks] if store_attn else []
        return logits, loss, attns


# -----------------------------------------------------------------------------
# Diversity regularizer
# -----------------------------------------------------------------------------


def attention_overlap_loss(
    attn_probs: torch.Tensor,
    causal_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross-head overlap penalty for one attention map [B, H, T, T].

    Uses the identity
        sum_{h != h'} <V_h, V_h'> = ||sum_h V_h||^2 - sum_h ||V_h||^2
    where V_h = A[b, h, i, :]. Divide by H*(H-1) and subtract 1/(i+1) so
    uniform causal attention yields 0. ``causal_mask`` is accepted for API
    completeness; the softmax output already has zeros on masked keys.
    """
    del causal_mask  # the masked-softmax input already zeros masked keys
    B, H, T, _ = attn_probs.shape
    assert H >= 2, "diversity loss requires at least 2 heads"
    sum_heads = attn_probs.sum(dim=1)                 # [B, T, T]
    sum_heads_sq = sum_heads.pow(2).sum(dim=-1)       # [B, T]
    sum_sq_heads = attn_probs.pow(2).sum(dim=(1, -1)) # [B, T]
    cross = sum_heads_sq - sum_sq_heads               # [B, T], ordered-pair sum
    avg_cross = cross / (H * (H - 1))
    idx = torch.arange(T, device=attn_probs.device, dtype=attn_probs.dtype)
    baseline = 1.0 / (idx + 1.0)                      # uniform causal overlap
    return (avg_cross - baseline[None, :]).mean()


def attention_overlap_loss_explicit(attn_probs: torch.Tensor) -> torch.Tensor:
    """Reference O(B*T*H^2) implementation used only for tests."""
    B, H, T, _ = attn_probs.shape
    total = attn_probs.new_zeros(())
    count = 0
    for b in range(B):
        for i in range(T):
            V = attn_probs[b, :, i, :]  # [H, T]
            s = attn_probs.new_zeros(())
            for h in range(H):
                for h2 in range(H):
                    if h == h2:
                        continue
                    s = s + (V[h] * V[h2]).sum()
            avg = s / (H * (H - 1))
            total = total + (avg - 1.0 / (i + 1.0))
            count += 1
    return total / count


def multi_layer_overlap_loss(attns: List[torch.Tensor]) -> torch.Tensor:
    if not attns:
        raise ValueError("no attention tensors given")
    return torch.stack([attention_overlap_loss(a) for a in attns]).mean()


# -----------------------------------------------------------------------------
# Diagnostics (no grad)
# -----------------------------------------------------------------------------


@torch.no_grad()
def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Per-head mean entropy (nats), returned as tensor of shape [H]."""
    eps = 1e-12
    ent = -(attn * (attn + eps).log()).sum(dim=-1)  # [B, H, T]
    return ent.mean(dim=(0, 2))


@torch.no_grad()
def top_k_jaccard(attn: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Mean Jaccard overlap of top-k attended keys across head pairs."""
    B, H, T, _ = attn.shape
    k = min(k, T)
    top = attn.topk(k=k, dim=-1).indices  # [B, H, T, k]
    sets = torch.zeros(B, H, T, T, device=attn.device, dtype=torch.bool)
    sets.scatter_(-1, top, True)
    totals = []
    for h in range(H):
        for h2 in range(h + 1, H):
            inter = (sets[:, h] & sets[:, h2]).sum(dim=-1).float()
            union = (sets[:, h] | sets[:, h2]).sum(dim=-1).float()
            j = torch.where(union > 0, inter / union, torch.zeros_like(union))
            totals.append(j.mean())
    return torch.stack(totals).mean() if totals else attn.new_zeros(())


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

SYNTHETIC_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "she sells sea shells by the sea shore. "
    "peter piper picked a peck of pickled peppers. "
    "how much wood would a woodchuck chuck if a woodchuck could chuck wood. "
) * 400


def load_text(path: Optional[str]) -> str:
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return SYNTHETIC_TEXT


def build_charset(text: str) -> Tuple[List[str], dict]:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return chars, stoi


def encode(text: str, stoi: dict) -> np.ndarray:
    return np.array([stoi[c] for c in text], dtype=np.int64)


def get_batch(
    data: np.ndarray, block_size: int, batch_size: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return (
        torch.from_numpy(x).to(device, non_blocking=True),
        torch.from_numpy(y).to(device, non_blocking=True),
    )


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


@dataclass
class RunResult:
    lambda_div: float
    train_lm: float
    val_lm: float
    diversity_loss: float
    mean_overlap: float
    mean_entropy: float
    mean_jaccard: float
    layer_overlaps: List[float]
    layer_entropies: List[float]
    note: str = ""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: MiniGPT, val: np.ndarray, cfg: GPTConfig, batches: int, device: str
) -> Tuple[float, float, List[float], List[float], float]:
    model.eval()
    lm_losses, div_losses, jaccards = [], [], []
    layer_ovs: List[List[float]] = [[] for _ in range(cfg.n_layer)]
    layer_ents: List[List[float]] = [[] for _ in range(cfg.n_layer)]
    with torch.no_grad():
        for _ in range(batches):
            x, y = get_batch(val, cfg.block_size, 16, device)
            _, loss, attns = model(x, y, store_attn=True)
            lm_losses.append(loss.item())
            div_losses.append(multi_layer_overlap_loss(attns).item())
            jacc = torch.stack([top_k_jaccard(a) for a in attns]).mean().item()
            jaccards.append(jacc)
            for li, a in enumerate(attns):
                layer_ovs[li].append(attention_overlap_loss(a).item())
                layer_ents[li].append(attention_entropy(a).mean().item())
    model.train()
    return (
        float(np.mean(lm_losses)),
        float(np.mean(div_losses)),
        [float(np.mean(x)) for x in layer_ovs],
        [float(np.mean(x)) for x in layer_ents],
        float(np.mean(jaccards)),
    )


def train_one(
    lambda_div: float,
    cfg: GPTConfig,
    train_data: np.ndarray,
    val_data: np.ndarray,
    steps: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
    log_every: int = 50,
) -> RunResult:
    set_seed(seed)
    model = MiniGPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    model.train()
    note = ""
    last_train = float("nan")
    last_div = float("nan")
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(train_data, cfg.block_size, batch_size, device)
        want_attn = lambda_div > 0
        _, lm_loss, attns = model(x, y, store_attn=want_attn)
        if want_attn:
            div_loss = multi_layer_overlap_loss(attns)
            loss = lm_loss + lambda_div * div_loss
            last_div = div_loss.item()
        else:
            loss = lm_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_train = lm_loss.item()
        if not math.isfinite(last_train):
            note = "train NaN"
            break
        if step == 1 or step % log_every == 0:
            dt = time.time() - t0
            print(
                f"[lambda={lambda_div:g}] step {step:4d} "
                f"lm={last_train:.4f} div={last_div:+.4f} t={dt:.1f}s"
            )
    val_lm, val_div, layer_ovs, layer_ents, jaccard = evaluate(
        model, val_data, cfg, batches=8, device=device
    )
    return RunResult(
        lambda_div=lambda_div,
        train_lm=last_train,
        val_lm=val_lm,
        diversity_loss=val_div,
        mean_overlap=float(np.mean(layer_ovs)),
        mean_entropy=float(np.mean(layer_ents)),
        mean_jaccard=jaccard,
        layer_overlaps=layer_ovs,
        layer_entropies=layer_ents,
        note=note,
    )


# -----------------------------------------------------------------------------
# Unit tests / sanity checks
# -----------------------------------------------------------------------------


def _uniform_causal_probs(B: int, H: int, T: int, device: str = "cpu") -> torch.Tensor:
    a = torch.zeros(B, H, T, T, device=device)
    for i in range(T):
        a[:, :, i, : i + 1] = 1.0 / (i + 1)
    return a


def run_unit_tests() -> None:
    torch.manual_seed(0)
    B, H, T = 2, 4, 8

    # 1) Uniform causal attention -> ~0
    uni = _uniform_causal_probs(B, H, T)
    loss_uni = attention_overlap_loss(uni).item()
    assert abs(loss_uni) < 1e-6, f"uniform should give 0, got {loss_uni}"

    # 2) Identical one-hot heads -> large positive loss = mean_i (1 - 1/(i+1))
    a = torch.zeros(B, H, T, T)
    for i in range(T):
        a[:, :, i, 0] = 1.0
    loss_id = attention_overlap_loss(a).item()
    expected_id = float(np.mean([1.0 - 1.0 / (i + 1) for i in range(T)]))
    assert abs(loss_id - expected_id) < 1e-5, (
        f"identical heads mismatch: {loss_id} vs {expected_id}"
    )

    # 3) Distinct one-hot heads -> lower than identical
    a = torch.zeros(B, H, T, T)
    for i in range(T):
        for h in range(H):
            pos = h % (i + 1)
            a[:, h, i, pos] = 1.0
    loss_diff = attention_overlap_loss(a).item()
    assert loss_diff < loss_id, (
        f"distinct heads should beat identical: {loss_diff} vs {loss_id}"
    )
    # As many positions as possible allow distinct keys -> overlap <= 0 there.
    # The i=0 row is unavoidable (only 1 valid key), pulling the mean above 0
    # only on that slot -- but the average should be close to zero or slightly
    # above.
    assert loss_diff <= loss_id

    # 4) Efficient vs explicit match on random softmax output
    torch.manual_seed(1)
    raw = torch.randn(B, H, T, T)
    raw = raw.masked_fill(
        ~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf")
    )
    a = torch.softmax(raw, dim=-1)
    eff = attention_overlap_loss(a).item()
    exp = attention_overlap_loss_explicit(a).item()
    assert abs(eff - exp) < 1e-5, f"efficient vs explicit mismatch {eff} vs {exp}"

    # 5) Gradient flows through efficient loss
    a_g = a.clone().requires_grad_(True)
    attention_overlap_loss(a_g).backward()
    assert a_g.grad is not None and torch.isfinite(a_g.grad).all()

    print("unit tests passed:")
    print(f"  uniform causal       loss = {loss_uni:+.6e}")
    print(f"  identical one-hot    loss = {loss_id:+.4f} (expected {expected_id:.4f})")
    print(f"  distinct  one-hot    loss = {loss_diff:+.4f}")
    print(f"  efficient vs explicit diff = {abs(eff - exp):.2e}")


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def print_table(results: List[RunResult]) -> None:
    print("\n=== Results ===")
    header = (
        f"{'lambda':>9} {'train_lm':>9} {'val_lm':>8} {'div':>10} "
        f"{'overlap':>9} {'entropy':>9} {'topk_jac':>9}  notes"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.lambda_div:>9.1e} {r.train_lm:>9.4f} {r.val_lm:>8.4f} "
            f"{r.diversity_loss:>+10.4f} {r.mean_overlap:>+9.4f} "
            f"{r.mean_entropy:>9.4f} {r.mean_jaccard:>9.4f}  {r.note}"
        )

    # Per-layer breakdown
    n_layer = len(results[0].layer_overlaps)
    print("\nPer-layer diagnostics:")
    for li in range(n_layer):
        print(f"  layer {li}:")
        for r in results:
            print(
                f"    lambda={r.lambda_div:.1e}  overlap={r.layer_overlaps[li]:+.4f}  "
                f"entropy={r.layer_entropies[li]:.4f}"
            )

    # Research summary
    print("\nResearch summary:")
    baseline = next((r for r in results if r.lambda_div == 0.0), results[0])
    best_val = min(results, key=lambda r: r.val_lm)
    most_div = min(results, key=lambda r: r.mean_overlap)
    print(
        f"  Baseline (lambda=0): val_lm={baseline.val_lm:.4f}, "
        f"overlap={baseline.mean_overlap:+.4f}, entropy={baseline.mean_entropy:.4f}"
    )
    print(
        f"  Best val_lm: lambda={best_val.lambda_div:g} -> "
        f"val_lm={best_val.val_lm:.4f} (delta={best_val.val_lm - baseline.val_lm:+.4f})"
    )
    print(
        f"  Most diverse: lambda={most_div.lambda_div:g} -> "
        f"overlap={most_div.mean_overlap:+.4f} "
        f"(delta={most_div.mean_overlap - baseline.mean_overlap:+.4f})"
    )
    print(
        "  Interpretation: increasing lambda should monotonically reduce overlap; "
        "val_lm typically stays flat at small lambda and rises once the penalty "
        "starts dominating. The best tradeoff is usually the largest lambda that "
        "leaves val_lm within noise of baseline."
    )


def make_plots(results: List[RunResult]) -> None:
    lambdas = [max(r.lambda_div, 1e-6) for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(lambdas, [r.val_lm for r in results], "o-")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("lambda_div")
    axes[0].set_ylabel("val LM loss")
    axes[0].set_title("Validation LM loss")
    axes[1].plot(lambdas, [r.mean_overlap for r in results], "o-")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("lambda_div")
    axes[1].set_ylabel("mean cross-head overlap")
    axes[1].set_title("Head overlap")
    axes[2].plot(lambdas, [r.mean_entropy for r in results], "o-")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("lambda_div")
    axes[2].set_ylabel("mean attention entropy (nats)")
    axes[2].set_title("Head entropy")
    fig.tight_layout()
    path = "head_diversity_sweep.png"
    fig.savefig(path, dpi=120)
    print(f"Saved plot to {path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=64)
    p.add_argument("--lambda-div", type=float, default=0.0)
    p.add_argument("--sweep", action="store_true",
                   help="run [0, 1e-4, 3e-4, 1e-3, 3e-3] instead of --lambda-div")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--data-path", type=str, default=None,
                   help="optional path to a local text file; uses synthetic text if missing")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--no-tests", action="store_true")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.no_tests:
        run_unit_tests()

    text = load_text(args.data_path)
    chars, stoi = build_charset(text)
    data = encode(text, stoi)
    split = int(0.9 * len(data))
    train, val = data[:split], data[split:]
    print(f"\nDataset: {len(data)} chars, vocab={len(chars)}, "
          f"source={'file:' + args.data_path if args.data_path and os.path.isfile(args.data_path) else 'synthetic'}")

    cfg = GPTConfig(
        vocab_size=len(chars),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=0.0,
    )

    lambdas = [0.0, 1e-4, 3e-4, 1e-3, 3e-3] if args.sweep else [args.lambda_div]

    results: List[RunResult] = []
    for ld in lambdas:
        print(f"\n=== Training lambda_div = {ld} ===")
        res = train_one(
            lambda_div=ld,
            cfg=cfg,
            train_data=train,
            val_data=val,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
        )
        results.append(res)

    print_table(results)
    if args.plot and HAVE_MPL:
        make_plots(results)
    elif args.plot:
        print("(matplotlib unavailable; skipping plot)")


if __name__ == "__main__":
    main()

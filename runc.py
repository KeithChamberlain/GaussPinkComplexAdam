"""Experiment: full 2x2 design (init x optimizer) in one invocation.

Task/benchmark: character-level LM on Python stdlib source.
- Metric 1: held-out cross-entropy loss (nats/char)
- Metric 2 (benchmark): held-out next-char top-1 accuracy

Usage: python runc.py N_REPEATS
Runs all 4 conditions with the SAME set of N unique seeds, guaranteeing
equal N per condition, duplicate seed.
Writes the 4 results_{init}_{opt}.json files analyze_orig.py expects as excel.
"""
import json, sys, time, glob
import torch
import torch.nn.functional as F
from model import CTransformerLM, ComplexAdam
from numpy.random import default_rng
import numpy as np
import pandas as pd
import openpyxl

SEQ, BATCH, STEPS, EVAL_BATCHES = 64, 16, 300, 20
D_MODEL, HEADS, LAYERS = 64, 2, 2
LR = 3e-3

INITS = ("gauss", "pink")
OPTS = ("torch_adam", "complex_adam")


def load_corpus():
    """Char corpus from Python stdlib source files (PSF-licensed, local)."""
    import sysconfig, os
    stdlib = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(stdlib, "*.py")))[:40]
    text = "".join(open(f, errors="ignore").read() for f in files)
    text = text[:400_000]
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def get_batch(data, gen):
    ix = torch.randint(len(data) - SEQ - 1, (BATCH,), generator=gen)
    x = torch.stack([data[i:i + SEQ] for i in ix])
    y = torch.stack([data[i + 1:i + SEQ + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model, data, gen):
    model.eval()
    losses, correct, total = [], 0, 0
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(data, gen)
        logits = model(x)
        losses.append(F.cross_entropy(logits.view(-1, logits.size(-1)),
                                      y.view(-1)).item())
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    model.train()
    return sum(losses) / len(losses), correct / total


def run_one(seed, init, opt_name, train, val, vocab, i):
    torch.manual_seed(seed)
    model = CTransformerLM(vocab, D_MODEL, HEADS, LAYERS, init=init)
    if opt_name == "torch_adam":
        opt = torch.optim.Adam(model.parameters(), lr=LR, foreach=False)
    elif opt_name == 'complex_adam':
        opt = ComplexAdam(model.parameters(), lr=LR)
    else: 
        pass
    gen = torch.Generator().manual_seed(seed)
    egen = torch.Generator().manual_seed(10_000 + seed)  # same eval batches per seed
    t0 = time.time()
    for step in range(STEPS):
        x, y = get_batch(train, gen)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    val_loss, val_acc = evaluate(model, val, egen)
    return dict(trial=i, seed=seed, init=init, opt=opt_name,
                val_loss=val_loss, val_acc=val_acc, secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    name = str(sys.argv[1])
    n = int(sys.argv[2])  # number of repeats per condition
    if n < 2:
        sys.exit("Need at least 2 repeats (analysis requires stdev).")
    if n > 12:
        print(f"WARNING: This will take a while.")
    rng = default_rng()
    rand_nums = rng.integers(0, 200_000, size=n, dtype=int) # unique seeds
    print("seeds:", rand_nums)

    train, val, vocab = load_corpus()
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    df = pd.DataFrame({"rows": [1], "n": [n]})
    df.to_excel(name)
    for init in INITS:
        for opt_name in OPTS:
            print(f"=== {init} / {opt_name} ===")
            out = []
            for i, s in enumerate(rand_nums):
                r = run_one(int(s), init, opt_name, train, val, vocab, i)
                out.append(r)
                print(json.dumps(r), flush=True)
            df = pd.DataFrame(out)
            with pd.ExcelWriter(name, engine='openpyxl', mode='a',
                if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=f'{init}_{opt_name}', index=False)

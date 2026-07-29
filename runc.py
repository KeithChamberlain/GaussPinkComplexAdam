"""Experiment: full 2x2 design (init x optimizer) in one invocation.

Task/benchmark: character-level LM on Python stdlib source.
- Metric 1: held-out cross-entropy loss (nats/char)
- Metric 2 (benchmark): held-out next-char top-1 accuracy

Usage: python runc.py OUT.xlsx N_REPEATS SEED
  OUT.xlsx    workbook to write (one sheet per condition)
  N_REPEATS   repeats per condition (>= 2)
  SEED        master seed for drawing per-run seeds; -1 for nondeterministic

Runs all 4 conditions with the SAME set of N unique seeds, guaranteeing
equal N per condition and a paired-by-seed design.

Weights: every run writes a checkpoint to CKPT_ROOT/<workbook stem>/ containing
the final model state_dict, the optimizer state, the config, and the metrics.
Initial weights are NOT saved -- they are exactly reproducible from
(seed, init) because torch.manual_seed(seed) is called immediately before
model construction and both inits draw only from the global RNG.
weights_analysis.py relies on that.
"""
import json, sys, time, glob
from pathlib import Path
import torch
import torch.nn.functional as F
from model import CTransformerLM, ComplexAdam
import numpy as np
import pandas as pd
import openpyxl
from numpy.random import Generator, PCG64DXSM
import logging


SEQ, BATCH, STEPS, EVAL_BATCHES = 64, 16, 300, 20
D_MODEL, HEADS, LAYERS = 64, 2, 2
LR = 3e-3

INITS = ("gauss", "pink")
OPTS = ("torch_adam", "complex_adam")

SAVE_WEIGHTS = True
SAVE_OPT_STATE = True      # needed for the second-moment anisotropy comparison
CKPT_ROOT = Path("checkpoints")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("InitilizerCrossOptimizerTest")
logger.setLevel(logging.INFO)


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


def config_dict(vocab):
    return dict(SEQ=SEQ, BATCH=BATCH, STEPS=STEPS, EVAL_BATCHES=EVAL_BATCHES,
                D_MODEL=D_MODEL, HEADS=HEADS, LAYERS=LAYERS, LR=LR, vocab=vocab)


def save_checkpoint(model, opt, seed, init, opt_name, vocab, metrics, ckpt_dir):
    """Final weights + optimizer state. ~0.6 MB at D_MODEL=64, LAYERS=2.

    state_dict() hands back live references, so detach/clone before writing --
    otherwise a later in-place optimizer step would mutate what you think is a
    snapshot. Everything is moved to CPU so checkpoints stay device-portable.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{init}__{opt_name}__seed{seed}.pt"
    payload = {
        "state_dict": {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()},
        "param_order": [n for n, _ in model.named_parameters()],
        "seed": int(seed), "init": init, "opt": opt_name,
        "config": config_dict(vocab),
        "metrics": metrics,
        "torch_version": torch.__version__,
    }
    if SAVE_OPT_STATE:
        # torch.optim.Adam keeps exp_avg_sq as a COMPLEX tensor (per-component
        # squares); ComplexAdam keeps v REAL (|g|^2). The dtype difference is
        # itself the evidence for the axis-alignment argument.
        payload["opt_state"] = opt.state_dict()
    torch.save(payload, path)
    return str(path)


def run_one(seed, init, opt_name, train, val, vocab, i, ckpt_dir=None):
    torch.manual_seed(seed)
    model = CTransformerLM(vocab, D_MODEL, HEADS, LAYERS, init=init)
    if opt_name == "torch_adam":
        opt = torch.optim.Adam(model.parameters(), lr=LR, foreach=False)
    elif opt_name == "complex_adam":
        opt = ComplexAdam(model.parameters(), lr=LR)
    else:
        raise ValueError(f"unknown optimizer {opt_name!r}")
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
    evaluate_timer = time.time()
    val_loss, val_acc = evaluate(model, val, egen)
    end_timer = time.time()

    rec = dict(
        trial=i, seed=seed, init=init, opt=opt_name,
        val_loss=val_loss, val_acc=val_acc, secs=end_timer - t0,
        train=evaluate_timer - t0,
        evaluate=end_timer - evaluate_timer
    )
    if SAVE_WEIGHTS and ckpt_dir is not None:
        rec["ckpt"] = save_checkpoint(
            model, opt, seed, init, opt_name, vocab,
            dict(val_loss=val_loss, val_acc=val_acc), ckpt_dir)
    return rec


if __name__ == "__main__":
    name = str(sys.argv[1])
    n = int(sys.argv[2])   # number of repeats per condition
    seed = int(sys.argv[3])  # master seed; -1 -> nondeterministic
    if n < 2:
        sys.exit("Need at least 2 repeats (analysis requires stdev).")
    if n > 12:
        logger.info("This will take a while.")

    rnd = PCG64DXSM() if seed == -1 else PCG64DXSM(seed)
    rng = Generator(rnd)
    # choice(replace=False) -- integers() samples WITH replacement, which can
    # hand back duplicate seeds and silently break the paired design.
    rand_nums = np.sort(rng.choice(200_000, size=n, replace=False))
    logger.info(f"seeds: {rand_nums.tolist()}")

    train, val, vocab = load_corpus()
    device = torch.device("cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    ckpt_dir = CKPT_ROOT / Path(name).stem
    if SAVE_WEIGHTS:
        logger.info(f"checkpoints -> {ckpt_dir}")

    # Config/provenance sheet. Deliberately left as the default name "Sheet1":
    # bootstrap_analysis_me.py concatenates every sheet except one called
    # "Sheet1", so renaming this would fold config rows into the OLS frame.
    meta = pd.DataFrame([dict(master_seed=seed, n=n, seeds=json.dumps(rand_nums.tolist()),
                              ckpt_dir=str(ckpt_dir), **config_dict(vocab))])
    meta.to_excel(name, index=False)

    for init in INITS:
        for opt_name in OPTS:
            logger.info(f"=== {init} / {opt_name} ===")
            out = []
            for i, s in enumerate(rand_nums):
                r = run_one(int(s), init, opt_name, train, val, vocab, i,
                            ckpt_dir=ckpt_dir)
                out.append(r)
                logger.info(json.dumps(r))
            df = pd.DataFrame(out)
            with pd.ExcelWriter(name, engine='openpyxl', mode='a',
                                if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=f'{init}_{opt_name}', index=False)

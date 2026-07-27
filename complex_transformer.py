"""
Minimal complex-valued transformer block (PyTorch, dtype=torch.cfloat).

Design choices:
- modReLU activation (magnitude-thresholded, phase-preserving)
- Complex RMSNorm (normalizes magnitude RMS, leaves phase untouched)
- Native rotary: positions are literal phase rotations e^{i*theta}
- Attention logits via Re(q · conj(k))  [swap for |.|**2 to get Born-rule interference]
- Wirtinger gradients: free — torch autograd on complex tensors already
  computes dL/d(conj z), which is the correct descent direction.
- Init: circularly-symmetric Gaussian by default, optional 1/f magnitude spectrum.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- init ----------

def circular_gaussian_(w: torch.Tensor, std: float):
    """Standard complex init: i.i.d. CN(0, std^2)."""
    with torch.no_grad():
        w.real.normal_(0, std / math.sqrt(2))
        w.imag.normal_(0, std / math.sqrt(2))
    return w


def pink_init_(w: torch.Tensor, std: float, alpha: float = 1.0):
    """1/f^alpha magnitude spectrum across fan-in, uniform phase.
    Complex-native analog of the pink-noise init idea."""
    with torch.no_grad():
        fan_in = w.shape[-1]
        f = torch.arange(1, fan_in + 1, dtype=torch.float32)
        mag = f.pow(-alpha / 2)                      # power ~ 1/f^alpha
        mag = mag / mag.pow(2).mean().sqrt() * std   # match target RMS
        phase = torch.rand_like(w.real) * (2 * math.pi)
        w.copy_(mag * torch.exp(1j * phase))
    return w


class CLinear(nn.Module):
    def __init__(self, d_in, d_out, init="gauss"):
        super().__init__()
        self.w = nn.Parameter(torch.empty(d_out, d_in, dtype=torch.cfloat))
        std = 1.0 / math.sqrt(d_in)
        (pink_init_ if init == "pink" else circular_gaussian_)(self.w, std)

    def forward(self, z):
        return F.linear(z, self.w)


# ---------- nonlinearity & norm ----------

def mod_relu(z, bias):
    """ReLU on magnitude, phase preserved: relu(|z| + b) * z/|z|."""
    mag = z.abs()
    scale = F.relu(mag + bias) / (mag + 1e-8)
    return z * scale


class CRMSNorm(nn.Module):
    """Normalize RMS of |z| across features; real gain only (phase untouched)."""
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, z):
        rms = z.abs().pow(2).mean(-1, keepdim=True).sqrt()
        return z / (rms + 1e-8) * self.g


# ---------- rotary, done honestly ----------

def rope(z, base=10000.0):
    """Positions as phase rotation. In complex land this is one multiply:
    z_t <- z_t * exp(i * t * omega). No pair-splitting gymnastics."""
    B, T, H, D = z.shape
    omega = base ** (-torch.arange(D, device=z.device) / D)      # (D,)
    t = torch.arange(T, device=z.device).float()                 # (T,)
    theta = torch.einsum("t,d->td", t, omega)                    # (T, D)
    return z * torch.exp(1j * theta)[None, :, None, :]


# ---------- attention ----------

class CAttention(nn.Module):
    def __init__(self, d_model, n_heads, score="real"):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.wq, self.wk, self.wv = (CLinear(d_model, d_model) for _ in range(3))
        self.wo = CLinear(d_model, d_model)
        self.score = score  # "real" -> Re(q conj(k));  "born" -> |q conj(k)|^2

    def forward(self, z, causal=True):
        B, T, _ = z.shape
        q = rope(self.wq(z).view(B, T, self.h, self.dh))
        k = rope(self.wk(z).view(B, T, self.h, self.dh))
        v = self.wv(z).view(B, T, self.h, self.dh)

        # Hermitian inner product: sum q * conj(k)
        s = torch.einsum("bthd,bshd->bhts", q, k.conj())
        logits = s.real if self.score == "real" else s.abs().pow(2)
        logits = logits / math.sqrt(self.dh)

        if causal:
            mask = torch.triu(torch.ones(T, T, device=z.device, dtype=torch.bool), 1)
            logits = logits.masked_fill(mask, float("-inf"))

        a = F.softmax(logits, dim=-1).to(v.real.dtype)   # real weights
        out = torch.einsum("bhts,bshd->bthd", a.to(torch.cfloat), v)
        return self.wo(out.reshape(B, T, -1))


# ---------- block & model ----------

class CBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4):
        super().__init__()
        self.n1, self.n2 = CRMSNorm(d_model), CRMSNorm(d_model)
        self.attn = CAttention(d_model, n_heads)
        self.up = CLinear(d_model, ff_mult * d_model)
        self.down = CLinear(ff_mult * d_model, d_model)
        self.b = nn.Parameter(torch.zeros(ff_mult * d_model))  # modReLU bias

    def forward(self, z):
        z = z + self.attn(self.n1(z))
        z = z + self.down(mod_relu(self.up(self.n2(z)), self.b))
        return z


class CTransformerLM(nn.Module):
    def __init__(self, vocab, d_model=256, n_heads=4, n_layers=4):
        super().__init__()
        self.emb = nn.Parameter(
            circular_gaussian_(torch.empty(vocab, d_model, dtype=torch.cfloat), 0.02))
        self.blocks = nn.ModuleList(CBlock(d_model, n_heads) for _ in range(n_layers))
        self.norm = CRMSNorm(d_model)
        self.head = CLinear(d_model, vocab)

    def forward(self, idx):
        z = self.emb[idx]
        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)
        # real logits for cross-entropy; Re() keeps interference effects,
        # .abs() would discard sign information
        return self.head(z).real


# ---------- optimizer note ----------
# torch.optim.Adam handles complex params but second moment defaults to
# per-component (real/imag) squares -> axis-aligned artifacts. For a
# rotation-invariant version, keep v = beta2*v + (1-beta2)*|g|^2 (real),
# and step: p -= lr * m / (sqrt(v) + eps), with m complex. ~15 lines to
# subclass; or pass foreach=False and patch _single_tensor_adam.


if __name__ == "__main__":
    torch.manual_seed(0)
    model = CTransformerLM(vocab=1000)
    x = torch.randint(0, 1000, (2, 32))
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, 1000), x.view(-1))
    loss.backward()   # Wirtinger gradients, courtesy of torch
    print(f"loss={loss.item():.3f}, params are complex: {model.emb.is_complex()}")

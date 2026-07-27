"""Complex transformer + rotation-invariant ComplexAdam."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- init ----------------

def circular_gaussian_(w, std):
    with torch.no_grad():
        w.real.normal_(0, std / math.sqrt(2))
        w.imag.normal_(0, std / math.sqrt(2))
    return w


def pink_init_(w, std, alpha=1.0):
    """1/f^alpha magnitude across fan-in, uniform phase, RMS matched to std."""
    with torch.no_grad():
        fan_in = w.shape[-1]
        f = torch.arange(1, fan_in + 1, dtype=torch.float32)
        mag = f.pow(-alpha / 2)
        mag = mag / mag.pow(2).mean().sqrt() * std
        phase = torch.rand(w.shape) * (2 * math.pi)
        w.copy_(mag * torch.exp(1j * phase))
    return w


INITS = {"gauss": circular_gaussian_, "pink": pink_init_}


class CLinear(nn.Module):
    def __init__(self, d_in, d_out, init="gauss"):
        super().__init__()
        self.w = nn.Parameter(torch.empty(d_out, d_in, dtype=torch.cfloat))
        INITS[init](self.w, 1.0 / math.sqrt(d_in))

    def forward(self, z):
        return F.linear(z, self.w)


def mod_relu(z, bias):
    mag = z.abs()
    return z * (F.relu(mag + bias) / (mag + 1e-8))


class CRMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))

    def forward(self, z):
        rms = z.abs().pow(2).mean(-1, keepdim=True).sqrt()
        return z / (rms + 1e-8) * self.g


def rope(z, base=10000.0):
    B, T, H, D = z.shape
    omega = base ** (-torch.arange(D, device=z.device) / D)
    theta = torch.einsum("t,d->td", torch.arange(T, device=z.device).float(), omega)
    return z * torch.exp(1j * theta)[None, :, None, :]


class CAttention(nn.Module):
    def __init__(self, d_model, n_heads, init="gauss"):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.wq = CLinear(d_model, d_model, init)
        self.wk = CLinear(d_model, d_model, init)
        self.wv = CLinear(d_model, d_model, init)
        self.wo = CLinear(d_model, d_model, init)

    def forward(self, z):
        B, T, _ = z.shape
        q = rope(self.wq(z).view(B, T, self.h, self.dh))
        k = rope(self.wk(z).view(B, T, self.h, self.dh))
        v = self.wv(z).view(B, T, self.h, self.dh)
        s = torch.einsum("bthd,bshd->bhts", q, k.conj()).real / math.sqrt(self.dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=z.device), 1)
        s = s.masked_fill(mask, float("-inf"))
        a = F.softmax(s, dim=-1)
        out = torch.einsum("bhts,bshd->bthd", a.to(torch.cfloat), v)
        return self.wo(out.reshape(B, T, -1))


class CBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=2, init="gauss"):
        super().__init__()
        self.n1, self.n2 = CRMSNorm(d_model), CRMSNorm(d_model)
        self.attn = CAttention(d_model, n_heads, init)
        self.up = CLinear(d_model, ff_mult * d_model, init)
        self.down = CLinear(ff_mult * d_model, d_model, init)
        self.b = nn.Parameter(torch.zeros(ff_mult * d_model))

    def forward(self, z):
        z = z + self.attn(self.n1(z))
        z = z + self.down(mod_relu(self.up(self.n2(z)), self.b))
        return z


class CTransformerLM(nn.Module):
    def __init__(self, vocab, d_model=64, n_heads=2, n_layers=2, init="gauss"):
        super().__init__()
        self.emb = nn.Parameter(
            INITS[init](torch.empty(vocab, d_model, dtype=torch.cfloat), 0.02))
        self.blocks = nn.ModuleList(
            CBlock(d_model, n_heads, init=init) for _ in range(n_layers))
        self.norm = CRMSNorm(d_model)
        self.head = CLinear(d_model, vocab, init)

    def forward(self, idx):
        z = self.emb[idx]
        for blk in self.blocks:
            z = blk(z)
        return self.head(self.norm(z)).real


# ---------------- optimizers ----------------

class ComplexAdam(torch.optim.Optimizer):
    """Adam with rotation-invariant second moment for complex params.

    v tracks |g|^2 (a real scalar per element) instead of separate squares
    for real/imag components. Under a global phase rotation g -> e^{i*phi} g,
    the update direction rotates identically -- component-wise Adam does not
    have this property (its preconditioner is axis-aligned in the complex
    plane). Real-valued params fall through to standard Adam behavior since
    |g|^2 == g^2 for real g.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, (b1, b2), eps = group["lr"], group["betas"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Wirtinger convention: torch stores dL/d(conj z); descend along it
                g = p.grad
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)          # complex (or real)
                    st["v"] = torch.zeros_like(p, dtype=torch.float32
                                               if p.is_complex() else p.dtype)
                st["t"] += 1
                t, m, v = st["t"], st["m"], st["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).add_((g.abs() ** 2).real if p.is_complex()
                                else g * g, alpha=1 - b2)
                mhat = m / (1 - b1 ** t)
                vhat = v / (1 - b2 ** t)
                denom = vhat.sqrt().add_(eps)
                p.add_(-(lr * mhat / denom.to(mhat.dtype)
                         if p.is_complex() else lr * mhat / denom))

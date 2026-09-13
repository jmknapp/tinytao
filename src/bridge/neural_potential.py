"""Tiny linear potentials on residue features. Diagnostic for which M looks load-bearing."""

from __future__ import annotations

import math

import torch


FEATURE_MODULI = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 24, 32)


def _features(n: torch.Tensor) -> torch.Tensor:
    """n: [B] int64 -> [B, F] float32."""
    logn = torch.log(n.float().clamp(min=1)).unsqueeze(-1)
    parts = [logn]
    for m in FEATURE_MODULI:
        r = (n % m).long()
        parts.append(torch.nn.functional.one_hot(r, num_classes=m).float())
    return torch.cat(parts, dim=-1)


def feature_dim() -> int:
    return 1 + sum(FEATURE_MODULI)


def _modulus_slices() -> dict[int, tuple[int, int]]:
    i = 1
    out: dict[int, tuple[int, int]] = {}
    for m in FEATURE_MODULI:
        out[m] = (i, i + m)
        i += m
    return out


def rank_moduli(energy: dict[str, float]) -> list[tuple[int, float]]:
    ranked = [(int(m), e) for m, e in energy.items()]
    ranked.sort(key=lambda kv: (-kv[1], kv[0]))
    return ranked


def residue_profile(w: torch.Tensor, m: int) -> dict[str, float]:
    """Mean |weight| per residue of n mod m, averaged over the population."""
    start, end = _modulus_slices()[m]
    chunk = w[:, start:end].abs().mean(dim=0)
    return {str(r): float(chunk[r]) for r in range(m)}


def train_population(
    ns: list[int],
    fs: list[int],
    *,
    pop: int = 256,
    steps: int = 1500,
    lr: float = 3e-3,
    seed: int = 20260909,
    log_every: int = 0,
    require_cuda: bool = False,
) -> dict:
    """Each net is a linear V. Push V(F(n)) < V(n) on the finite odd sample."""
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this run")
    if not torch.cuda.is_available():
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    x = torch.tensor(ns, dtype=torch.int64, device=device)
    y = torch.tensor(fs, dtype=torch.int64, device=device)
    phi_x = _features(x)
    phi_y = _features(y)
    Fdim = phi_x.shape[1]
    W = torch.empty(pop, Fdim, device=device, requires_grad=True)
    b = torch.zeros(pop, device=device, requires_grad=True)
    with torch.no_grad():
        W.normal_(0.0, 1.0 / math.sqrt(Fdim))
    opt = torch.optim.Adam([W, b], lr=lr)
    last = {}
    history = []
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        vx = torch.einsum("bf,pf->pb", phi_x, W) + b[:, None]
        vy = torch.einsum("bf,pf->pb", phi_y, W) + b[:, None]
        loss = torch.relu(vy - vx + 0.1).mean()
        loss.backward()
        opt.step()
        if log_every and (step == 1 or step % log_every == 0 or step == steps):
            frac = float((vy < vx).float().mean().detach())
            rec = {"step": step, "loss": float(loss.detach()), "frac_decrease": frac}
            history.append(rec)
            print(f"step {step:5d}  loss={rec['loss']:.4f}  V(F)<V(n)={frac:.4f}", flush=True)
        if step == steps:
            last = {
                "loss": float(loss.detach()),
                "frac_decrease": float((vy < vx).float().mean().detach()),
            }
    with torch.no_grad():
        w = W.detach().cpu()
        energy = {}
        for m, (i, j) in _modulus_slices().items():
            energy[str(m)] = float(w[:, i:j].pow(2).mean())
        last["modulus_weight_energy"] = energy
        last["logn_energy"] = float(w[:, 0].pow(2).mean())
        last["ranked_moduli"] = [{"M": m, "energy": e} for m, e in rank_moduli(energy)]
        top_m = rank_moduli(energy)[0][0]
        last["top_M"] = top_m
        last["top_M_residue_profile"] = residue_profile(w, top_m)
        last["pop"] = pop
        last["steps"] = steps
        last["device"] = str(device)
        last["history"] = history
    return last

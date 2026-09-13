"""Affine lemma heads on the locked feature bank. No energy shortlist.

Each (M, r, k) is a tiny population of maps q |-> A q + B. After Adam, A and B
are rounded to integers. A residue is covered only if some head matches F^k
exactly on the train class and descends. Misses may use exact two-point affine
leftover (15.5) then odd-part fill.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from src.bridge.blackbox import BlackBoxMap
from src.bridge.discoverer import (
    ProposedCertificate,
    ProposedLemma,
    _fit_odd_part,
    _samples,
    _try_affine,
    fill_uncovered,
)
from src.bridge.neural_potential import FEATURE_MODULI


K_MAX = 4


def _residues(M: int) -> tuple[int, ...]:
    if M % 2 == 1:
        return tuple(range(M))
    return tuple(range(1, M, 2))


def _precompute_fk(bb: BlackBoxMap, ns: list[int], k_max: int) -> list[list[int]]:
    out = []
    for n in ns:
        row = []
        x = n
        for _ in range(k_max):
            x = bb.F_odd(x)
            row.append(x)
        out.append(row)
    return out


def train_affine_heads(
    bb: BlackBoxMap,
    ns: list[int],
    M: int,
    *,
    pop: int = 256,
    steps: int = 800,
    lr: float = 5e-2,
    seed: int = 20260909,
    k_max: int = K_MAX,
    require_cuda: bool = False,
    log_every: int = 0,
) -> dict:
    """Learn A,B per (residue, k) on one modulus. CUDA vectorized over the population."""
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed + M)
    residues = _residues(M)
    r_to_i = {r: i for i, r in enumerate(residues)}
    R, K = len(residues), k_max
    fk = _precompute_fk(bb, ns, K)
    n_t = torch.tensor(ns, dtype=torch.int64, device=device)
    q_t = torch.div(n_t, M, rounding_mode="floor")
    r_t = n_t % M
    idx = torch.tensor(
        [r_to_i.get(int(r), -1) for r in r_t.tolist()],
        dtype=torch.int64,
        device=device,
    )
    mask = idx >= 0
    y = torch.tensor(fk, dtype=torch.float32, device=device)
    n_f = n_t.float()
    q_f = q_t.float()
    A = torch.empty(pop, R, K, device=device, requires_grad=True)
    B = torch.empty(pop, R, K, device=device, requires_grad=True)
    with torch.no_grad():
        A.normal_(0.0, 1.0)
        B.normal_(0.0, 1.0)
    opt = torch.optim.Adam([A, B], lr=lr)
    last_loss = float("nan")
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        A_g = A[:, idx.clamp(min=0), :]
        B_g = B[:, idx.clamp(min=0), :]
        pred = A_g * q_f[None, :, None] + B_g
        mse = (pred - y[None, :, :]) ** 2
        desc = torch.relu(pred - n_f[None, :, None] + 0.1)
        w = mask.float()[None, :, None]
        denom = (w.sum() * K * pop).clamp(min=1.0)
        loss = ((mse + desc) * w).sum() / denom
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())
        if log_every and (step == 1 or step % log_every == 0 or step == steps):
            print(f"  M={M:<3} step {step:4d}  loss={last_loss:.4f}", flush=True)
    with torch.no_grad():
        A_i = A.detach().round().long()
        B_i = B.detach().round().long()
        A_g = A_i[:, idx.clamp(min=0), :]
        B_g = B_i[:, idx.clamp(min=0), :]
        pred_i = A_g * q_t[None, :, None] + B_g
        y_i = torch.tensor(fk, dtype=torch.int64, device=device)
        exact = (pred_i == y_i[None, :, :]) & mask[None, :, None]
        descends = (pred_i < n_t[None, :, None]) & mask[None, :, None]
    return {
        "M": M,
        "residues": residues,
        "A": A_i.cpu(),
        "B": B_i.cpu(),
        "exact": exact.cpu(),
        "descends": descends.cpu(),
        "idx": idx.cpu(),
        "mask": mask.cpu(),
        "q": q_t.cpu(),
        "n": n_t.cpu(),
        "loss": last_loss,
        "device": str(device),
        "pop": pop,
        "steps": steps,
        "k_max": K,
    }


def _lemma_from_heads(fit: dict, r: int) -> ProposedLemma | None:
    """Best exact descending affine head for residue r, else None."""
    residues: tuple[int, ...] = fit["residues"]
    ri = residues.index(r)
    M = fit["M"]
    idx = fit["idx"]
    mask_r = (idx == ri) & fit["mask"]
    n_train = int(mask_r.sum())
    if n_train < 3:
        return None
    exact = fit["exact"][:, mask_r, :]
    desc = fit["descends"][:, mask_r, :]
    all_exact = exact.all(dim=1)
    all_desc = desc.all(dim=1)
    ok = all_exact & all_desc
    A = fit["A"][:, ri, :]
    B = fit["B"][:, ri, :]
    for k in range(1, fit["k_max"] + 1):
        ki = k - 1
        hits = torch.nonzero(ok[:, ki] & (A[:, ki] != 0), as_tuple=False).flatten()
        if hits.numel() == 0:
            continue
        p = int(hits[0])
        return ProposedLemma(
            M=M,
            r=r,
            k=k,
            A=int(A[p, ki]),
            B=int(B[p, ki]),
            n_train=n_train,
            source="neural_round",
        )
    return None


def certificate_from_heads(
    bb: BlackBoxMap,
    fit: dict,
    n_max: int,
    *,
    exact_affine_leftover: bool = False,
) -> ProposedCertificate:
    M = fit["M"]
    lemmas: list[ProposedLemma] = []
    uncovered: list[int] = []
    k_max = int(fit["k_max"])
    for r in fit["residues"]:
        found = _lemma_from_heads(fit, r)
        if found is None and exact_affine_leftover:
            pts = _samples(M, r, n_max)
            for k in range(1, k_max + 1):
                found = _try_affine(bb, M, r, pts, k=k)
                if found is not None:
                    found = replace(found, source="exact_affine")
                    break
        if found is None:
            found = _fit_odd_part(bb, M, r, n_max)
            if found is not None:
                found = replace(found, source="odd_part")
        if found is None:
            if _samples(M, r, n_max):
                uncovered.append(r)
        else:
            lemmas.append(found)
    n_res = len(fit["residues"])
    cover = len(lemmas) / max(n_res, 1)
    cert = ProposedCertificate(
        map_name=bb.name,
        M=M,
        k_max=fit["k_max"],
        n_max_train=n_max,
        lemmas=tuple(lemmas),
        uncovered=tuple(uncovered),
        train_cover_frac=cover,
    )
    return fill_uncovered(bb, cert, n_max)


def propose_neural_certificates(
    bb: BlackBoxMap,
    ns: list[int],
    *,
    moduli: tuple[int, ...] = FEATURE_MODULI,
    pop: int = 256,
    steps: int = 800,
    lr: float = 5e-2,
    seed: int = 20260909,
    k_max: int = K_MAX,
    require_cuda: bool = False,
    log_every: int = 0,
    exact_affine_leftover: bool = False,
) -> tuple[list[ProposedCertificate], list[dict]]:
    """One certificate per bank M, from rounded affine heads + leftover fill."""
    n_max = max(ns)
    certs = []
    fits = []
    for M in moduli:
        print(f"neural lemmas M={M}", flush=True)
        fit = train_affine_heads(
            bb, ns, M,
            pop=pop, steps=steps, lr=lr, seed=seed, k_max=k_max,
            require_cuda=require_cuda, log_every=log_every,
        )
        cert = certificate_from_heads(
            bb, fit, n_max, exact_affine_leftover=exact_affine_leftover,
        )
        n_aff = sum(1 for L in cert.lemmas if L.kind == "affine")
        n_neural = sum(1 for L in cert.lemmas if L.source == "neural_round")
        n_exact = sum(1 for L in cert.lemmas if L.source == "exact_affine")
        print(
            f"  cover={cert.train_cover_frac:.3f}  affine={n_aff}  "
            f"neural_round={n_neural}  exact_affine={n_exact}  "
            f"odd_part={len(cert.lemmas) - n_aff}  uncovered={list(cert.uncovered)}  "
            f"loss={fit['loss']:.4f}",
            flush=True,
        )
        certs.append(cert)
        fits.append(
            {
                "M": M,
                "loss": fit["loss"],
                "cover": cert.train_cover_frac,
                "uncovered": list(cert.uncovered),
                "n_affine": n_aff,
                "n_neural_round": n_neural,
                "n_exact_affine": n_exact,
                "n_odd_part": len(cert.lemmas) - n_aff,
            }
        )
    certs.sort(
        key=lambda c: (
            -c.train_cover_frac,
            0 if not c.uncovered else 1,
            c.M,
            max((L.k for L in c.lemmas), default=99),
        )
    )
    return certs, fits

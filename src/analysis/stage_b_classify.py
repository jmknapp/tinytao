"""Stage B exact-net classification: Family F, Family Q, table homomorphism.

Every exact network starts as EXACT-UNCLASSIFIED. Family F/Q require 100%
neural-free replacement. Do not classify by R² alone.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.analysis.class2_mod4 import certify_class2_net
from src.analysis.control_tables import genuine_mul, inverse_perm
from src.analysis.fourier import discrete_log_table
from src.analysis.group_law import circ_mean, fit_dlog_circle_batch, wrap
from src.analysis.level4_h1 import effective_winding
from src.analysis.stats import wilson_interval
from src.analysis.torus_crt import certify_torus_net, try_family_q_on_head
from src.symbolic_f13 import classify_pairs, from_fit

DESTROYED_LAW = frozenset(
    {"D_random_comm", "E_latin_nonassoc", "E_latin_comm_nonassoc", "F_assoc_damage"}
)
CYCLIC_GROUPISH = frozenset({"genuine", "A_isomorphic", "B_output_perm"})


def remap_dlog(p: int, pi: np.ndarray | None) -> np.ndarray:
    dlog = discrete_log_table(p)
    if pi is None:
        return dlog
    out = np.full(p, -1, dtype=np.int64)
    for i, src in enumerate(pi):
        out[i + 1] = int(dlog[int(src) + 1])
    return out


def reorder_to_abstract(xy: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """xy_abs[π(i)] = xy_disp[i]."""
    inv = inverse_perm(np.asarray(pi, dtype=np.int64))
    return xy[inv]


def table_homomorphism_delta(xy: np.ndarray, table: np.ndarray) -> np.ndarray:
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    return wrap(theta[table] - theta[:, None] - theta[None, :])


def summarize_delta(delta: np.ndarray) -> dict[str, float]:
    mu = float(circ_mean(delta.reshape(-1)))
    centered = wrap(delta - mu)
    abs_c = np.abs(centered)
    return {
        "circ_mean_rad": mu,
        "mean_abs_centered_rad": float(abs_c.mean()),
        "mean_abs_centered_deg": float(np.degrees(abs_c.mean())),
        "max_abs_centered_deg": float(np.degrees(abs_c.max())),
        "rms_centered_deg": float(np.degrees(np.sqrt((centered**2).mean()))),
    }


def composition_nearest_decoder(xy: np.ndarray, table: np.ndarray) -> dict[str, Any]:
    """Predict a★b as the embedding nearest to z(a)z(b). Neural-free."""
    z = xy[:, 0] + 1j * xy[:, 1]
    h = z[:, None] * z[None, :]
    dist = np.abs(h[..., None] - z[None, None, :])
    pred = dist.argmin(axis=-1)
    n_ok = int((pred == table).sum())
    n = int(table.size)
    return {
        "acc": n_ok / n,
        "exact": bool(n_ok == n),
        "n_correct": n_ok,
        "n_pairs": n,
    }


def radius_cv(xy: np.ndarray) -> float:
    r = np.linalg.norm(xy, axis=-1)
    return float(r.std() / (r.mean() + 1e-12))


def classify_exact_embeddings(
    xy: np.ndarray,
    table: np.ndarray,
    p: int,
    control: str,
    pi: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """xy [P, n, 2] embeddings of exact nets in displayed-symbol order.

    Locked H=1 Family F / Family Q path. For H>=2 tensors [P,n,H,2],
    use classify_exact_multihead (or pass ndim==4 here to dispatch).
    """
    if getattr(xy, "ndim", 0) == 4:
        return classify_exact_multihead(xy, table, p, control, pi=pi, rng=rng)

    n_saved = int(xy.shape[0])
    n = p - 1
    rng = np.random.default_rng(p) if rng is None else rng
    a = np.arange(1, p).repeat(n)
    b = np.tile(np.arange(1, p), n)
    y = table.reshape(-1)

    dlog_fit = remap_dlog(p, pi if control == "A_isomorphic" else None)
    analysis_xy = xy
    if control == "A_isomorphic" and pi is not None and n_saved:
        analysis_xy = np.stack([reorder_to_abstract(xy[i], pi) for i in range(n_saved)], axis=0)
        dlog_fit = discrete_log_table(p)

    n_faithful = 0
    n_class2 = 0
    n_comp_exact = 0
    n_unclassified = 0
    n_fpstar_q_geometry = 0
    gcd_hist: dict[int, int] = {}
    hom_deg = []
    comp_accs = []
    radius_cvs = []
    novel = []

    batch = None
    k_eff = None
    if n_saved:
        batch = fit_dlog_circle_batch(analysis_xy, dlog_fit, n)
        k_eff = effective_winding(batch["k"], batch["conjugate"], n)

    for i in range(n_saved):
        label = "EXACT-UNCLASSIFIED"
        ki = int(k_eff[i]) if k_eff is not None else 0
        gcd = math.gcd(ki, n) if ki else n
        gcd_hist[gcd] = gcd_hist.get(gcd, 0) + 1
        rcv = radius_cv(analysis_xy[i])
        radius_cvs.append(rcv)
        delta = table_homomorphism_delta(xy[i], table)
        hom = summarize_delta(delta)
        hom_deg.append(hom["mean_abs_centered_deg"])
        comp = composition_nearest_decoder(xy[i], table)
        comp_accs.append(comp["acc"])
        if comp["exact"]:
            n_comp_exact += 1

        faithful = False
        cert_break = None
        if batch is not None and gcd == 1:
            fit = {
                "k": int(batch["k"][i]),
                "phi": float(batch["phi"][i]),
                "scale": float(batch["scale"][i]),
                "conjugate": bool(batch["conjugate"][i]),
            }
            pred = classify_pairs(from_fit(p, fit), a, b, dlog_fit)
            if control == "B_output_perm" and pi is not None:
                pred = np.asarray(pi, dtype=np.int64)[pred]
                faithful = bool((pred == y).all())
            elif control == "A_isomorphic":
                faithful = bool((pred == genuine_mul(p).reshape(-1)).all())
            else:
                faithful = bool((pred == y).all())
            if faithful:
                n_faithful += 1
                label = "FAMILY_F"
        if batch is not None and gcd == 2 and label == "EXACT-UNCLASSIFIED":
            cert = certify_class2_net(
                analysis_xy[i], ki, float(batch["phi"][i]), p, rng
            )
            # Family Q is a certificate for the cyclic group law, not for a
            # scrambled table. The decoder reconstructs F_p* multiplication
            # (or its isomorphic copy). Counting it on destroyed-law tables
            # would credit residual F_p* geometry rather than solving ★.
            if cert["certified"] and control in ("genuine", "A_isomorphic", "B_output_perm"):
                n_class2 += 1
                label = "FAMILY_Q"
            elif cert["certified"]:
                n_fpstar_q_geometry += 1
            cert_break = cert.get("break_at")

        if label == "EXACT-UNCLASSIFIED":
            n_unclassified += 1
            if comp["exact"] and control in DESTROYED_LAW:
                novel.append(
                    {
                        "net": i,
                        "hom_deg": hom["mean_abs_centered_deg"],
                        "radius_cv": rcv,
                        "gcd": gcd,
                        "cert_break": cert_break,
                    }
                )
                label = "NOVEL_COMPOSITION"

    n_other_symbolic = int(len(novel))
    return {
        "n_saved": n_saved,
        "n_faithful": n_faithful,
        "n_class2": n_class2,
        "n_other_symbolic": n_other_symbolic,
        "n_unclassified": n_unclassified - n_other_symbolic,
        "n_composition_exact": n_comp_exact,
        "n_fpstar_q_geometry": n_fpstar_q_geometry,
        "gcd_hist": {str(k): v for k, v in sorted(gcd_hist.items())},
        "mean_hom_deg": float(np.mean(hom_deg)) if hom_deg else None,
        "median_hom_deg": float(np.median(hom_deg)) if hom_deg else None,
        "mean_composition_acc": float(np.mean(comp_accs)) if comp_accs else None,
        "mean_radius_cv": float(np.mean(radius_cvs)) if radius_cvs else None,
        "novel_composition_nets": novel[:32],
        "n_novel_composition": n_other_symbolic,
    }


def classify_exact_multihead(
    E: np.ndarray,
    table: np.ndarray,
    p: int,
    control: str,
    pi: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Classify exact multi-head nets. E: [P, n, H, 2], H>=2.

    Does not alter the locked H=1 Family F/Q census. Emits TORUS_Tm
    (and FAMILY_F / FAMILY_Q when a single selected head carries F or Q).
    """
    E = np.asarray(E, dtype=np.float64)
    if E.ndim != 4 or E.shape[-1] != 2:
        raise ValueError(f"expected E [P,n,H,2], got {E.shape}")
    n_saved, n, heads, _ = E.shape
    if n != p - 1:
        raise ValueError(f"E.shape[1]={n} != p-1={p-1}")
    if heads < 2:
        # Fall back to H=1 locked path on head 0
        return classify_exact_embeddings(E[:, :, 0, :], table, p, control, pi=pi, rng=rng)

    rng = np.random.default_rng(p) if rng is None else rng
    analysis_E = E
    if control == "A_isomorphic" and pi is not None and n_saved:
        inv = inverse_perm(np.asarray(pi, dtype=np.int64))
        analysis_E = E[:, inv, :, :]

    torus_hist: dict[str, int] = {}
    n_torus = 0
    n_faithful = 0
    n_class2 = 0
    n_unclassified = 0
    labels: list[str] = []
    details: list[dict[str, Any]] = []

    for i in range(n_saved):
        cert = certify_torus_net(analysis_E[i], p)
        label = "EXACT-UNCLASSIFIED"
        if cert["certified"]:
            n_torus += 1
            m = int(cert["m"])
            label = f"TORUS_T{m}"
            # Single surviving factor: inherit locked F/Q names when applicable
            if m == 1:
                sel = cert["selected_heads"][0]
                fit = cert["fits"][sel]
                if int(fit["gcd"]) == 1:
                    label = "FAMILY_F"
                    n_faithful += 1
                elif int(fit["gcd"]) == 2 and control in CYCLIC_GROUPISH:
                    q = try_family_q_on_head(analysis_E[i, :, sel, :], fit, p)
                    if q.get("certified"):
                        label = "FAMILY_Q"
                        n_class2 += 1
                    else:
                        label = "TORUS_T1"
            torus_hist[f"T{m}"] = torus_hist.get(f"T{m}", 0) + 1
        else:
            n_unclassified += 1
        labels.append(label)
        details.append(
            {
                "net": i,
                "label": label,
                "m": cert.get("m"),
                "lcm": cert.get("lcm"),
                "break_at": cert.get("break_at"),
                "decoder_exact": cert.get("decoder_exact"),
                "ablation_ok": cert.get("ablation_ok"),
                "selected_heads": cert.get("selected_heads"),
            }
        )

    return {
        "n_saved": n_saved,
        "heads": heads,
        "n_torus_certified": n_torus,
        "n_faithful": n_faithful,
        "n_class2": n_class2,
        "n_unclassified": n_unclassified,
        "torus_hist": torus_hist,
        "labels": labels,
        "details": details[:64],
    }


def baseline_inconsistent(p: int, n_exact: int, n_pop: int, n_class2: int, prior: dict) -> tuple[bool, str]:
    rate = n_exact / max(n_pop, 1)
    prior_rate = prior["exact"] / 1024
    if n_exact == 0:
        return True, "exact_count_is_zero"
    if abs(rate - prior_rate) > 0.20:
        return True, f"exact_rate_diff {rate:.3f} vs prior {prior_rate:.3f} > 0.20"
    if p in (11, 19) and n_class2 == 0:
        return True, "class2_count_is_zero_on_p_equiv_3_mod4"
    if p == 29 and n_class2 > 0:
        return True, "class2_count_positive_on_p_equiv_1_mod4"
    return False, "ok"


def family_summary(rows: list[dict]) -> list[dict]:
    from collections import defaultdict

    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        buckets[r["control"]].append(r)
    out = []
    for ctrl, xs in sorted(buckets.items()):
        n_pop = sum(int(x["population"]) for x in xs)
        n_exact = sum(int(x["n_exact"]) for x in xs)
        n_f = sum(int(x["n_faithful"]) for x in xs)
        n_q = sum(int(x["n_class2"]) for x in xs)
        n_o = sum(int(x["n_other_symbolic"]) for x in xs)
        n_geo = sum(int(x.get("n_fpstar_q_geometry") or (x.get("classification") or {}).get("n_fpstar_q_geometry") or 0) for x in xs)
        pe, lo, hi = wilson_interval(n_exact, n_pop) if n_pop else (0, 0, 0)
        out.append(
            {
                "control": ctrl,
                "n_jobs": len(xs),
                "n_pop": n_pop,
                "n_exact": n_exact,
                "n_faithful": n_f,
                "n_class2": n_q,
                "n_other_symbolic": n_o,
                "n_fpstar_q_geometry": n_geo,
                "p_exact": pe,
                "p_exact_lo": lo,
                "p_exact_hi": hi,
                "mean_train_acc": float(np.mean([x["mean_train_acc"] for x in xs])),
                "mean_held_acc": float(np.mean([x["mean_test_acc"] for x in xs])),
                "mean_domain_acc": float(np.mean([x["mean_domain_acc"] for x in xs])),
                "mean_associativity": float(np.mean([x["associativity_fraction"] for x in xs])),
            }
        )
    return out

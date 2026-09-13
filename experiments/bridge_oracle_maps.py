"""Build hidden oracle certificates. Do not train. Discoverer must not read this tree."""

from __future__ import annotations

import json
from pathlib import Path

from src.bridge.catalog import CANDIDATE_MAPS
from src.bridge.oracle import certificate_to_json, prove_eventual_descent


def main() -> None:
    out = Path("tiny_tao_results/bridge_oracle_hidden")
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for factory in CANDIDATE_MAPS:
        m = factory()
        print("proving", m.name, flush=True)
        cert = prove_eventual_descent(m, max_horizon=12, small_upto=512)
        payload = {
            "name": m.name,
            "halt": m.halt,
            "rule_modulus": m.modulus,
            "rules": [
                {"r": r.r, "m": r.m, "a": r.a, "b": r.b} for r in m.rules
            ],
            "certificate": certificate_to_json(cert),
        }
        path = out / f"{m.name}.json"
        path.write_text(json.dumps(payload, indent=2))
        index.append(
            {
                "name": m.name,
                "verified": cert.verified,
                "modulus": cert.modulus,
                "n_lemmas": len(cert.lemmas),
                "n_failures": len(cert.failures),
                "horizons": sorted({L.k for L in cert.lemmas}),
                "path": str(path),
            }
        )
        print(m.name, "verified=", cert.verified, "M=", cert.modulus, "k=", sorted({L.k for L in cert.lemmas}))
        if cert.failures[:8]:
            print("  failures:", cert.failures[:8])
    (out / "index.json").write_text(json.dumps(index, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()

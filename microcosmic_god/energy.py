from __future__ import annotations

ENERGY_KINDS = (
    "solar",
    "essence",
    "residue_store",
    "thermal",
    "mechanical",
    "electrical",
    "dense_node",
)


def blank_energy(value: float = 0.0) -> dict[str, float]:
    return {kind: float(value) for kind in ENERGY_KINDS}

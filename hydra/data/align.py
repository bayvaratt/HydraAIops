"""Feature alignment layer for cross-dataset generalisation.

Derives a 6-feature canonical representation that is computable from both
TON_IoT and CIC-IoT-2023, enabling train-on-A / evaluate-on-B experiments.
"""
from __future__ import annotations

import pandas as pd

CANONICAL_FEATURES = [
    "f_total_pkts",
    "f_total_bytes",
    "f_duration",
    "f_is_tcp",
    "f_is_udp",
    "f_is_icmp",
]


def align_ton_iot(df: pd.DataFrame) -> pd.DataFrame:
    """Map TON_IoT columns → canonical features + label + type."""
    out = pd.DataFrame(index=df.index)
    out["f_total_pkts"]  = df["src_pkts"].fillna(0) + df["dst_pkts"].fillna(0)
    out["f_total_bytes"] = df["src_bytes"].fillna(0) + df["dst_bytes"].fillna(0)
    out["f_duration"]    = df["duration"]
    proto = df["proto"].fillna("").astype(str).str.lower()
    out["f_is_tcp"]  = (proto == "tcp").astype(int)
    out["f_is_udp"]  = (proto == "udp").astype(int)
    out["f_is_icmp"] = (proto == "icmp").astype(int)
    out["label"] = df["label"]
    out["type"]  = df["type"]
    return out


def align_cic_iot2023(df: pd.DataFrame) -> pd.DataFrame:
    """Map CIC-IoT-2023 columns → canonical features + label + type."""
    out = pd.DataFrame(index=df.index)
    out["f_total_pkts"]  = df["number"]
    out["f_total_bytes"] = df["tot_size"]
    out["f_duration"]    = df["iat"]
    out["f_is_tcp"]  = df["tcp"].fillna(0).astype(int)
    out["f_is_udp"]  = df["udp"].fillna(0).astype(int)
    out["f_is_icmp"] = df["icmp"].fillna(0).astype(int)
    out["label"] = df["label"]
    out["type"]  = df["type"]
    return out


ALIGNERS: dict[str, callable] = {
    "ton_iot": align_ton_iot,
    "ton_iot_dedup": align_ton_iot,
    "cic_iot2023": align_cic_iot2023,
}

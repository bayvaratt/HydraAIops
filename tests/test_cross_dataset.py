"""Tests for hydra.data.align — cross-dataset canonical feature alignment."""
import numpy as np
import pandas as pd
import pytest

from hydra.data.align import align_ton_iot, align_cic_iot2023, CANONICAL_FEATURES, ALIGNERS


def test_canonical_features_count():
    assert len(CANONICAL_FEATURES) == 6


def test_canonical_features_names():
    expected = {"f_total_pkts", "f_total_bytes", "f_duration", "f_is_tcp", "f_is_udp", "f_is_icmp"}
    assert set(CANONICAL_FEATURES) == expected


def test_align_ton_iot_output_columns():
    df = pd.DataFrame({
        "src_pkts": [10, 20], "dst_pkts": [5, 10],
        "src_bytes": [100, 200], "dst_bytes": [50, 100],
        "duration": [1.0, 2.0], "proto": ["tcp", "udp"],
        "label": [0, 1], "type": ["normal", "dos"],
    })
    result = align_ton_iot(df)
    assert list(result.columns) == CANONICAL_FEATURES + ["label", "type"]
    assert result["f_total_pkts"].tolist() == [15, 30]
    assert result["f_total_bytes"].tolist() == [150, 300]
    assert result["f_is_tcp"].tolist() == [1, 0]
    assert result["f_is_udp"].tolist() == [0, 1]


def test_align_ton_iot_icmp():
    df = pd.DataFrame({
        "src_pkts": [5], "dst_pkts": [3],
        "src_bytes": [50], "dst_bytes": [30],
        "duration": [0.5], "proto": ["icmp"],
        "label": [1], "type": ["scan"],
    })
    result = align_ton_iot(df)
    assert result["f_is_icmp"].tolist() == [1]
    assert result["f_is_tcp"].tolist() == [0]
    assert result["f_is_udp"].tolist() == [0]


def test_align_cic_iot2023_output_columns():
    df = pd.DataFrame({
        "number": [10, 20], "tot_size": [100, 200],
        "iat": [1.0, 2.0], "tcp": [1, 0], "udp": [0, 1], "icmp": [0, 0],
        "label": [0, 1], "type": ["normal", "ddos"],
    })
    result = align_cic_iot2023(df)
    assert list(result.columns) == CANONICAL_FEATURES + ["label", "type"]
    assert result["f_total_pkts"].tolist() == [10, 20]


def test_align_cic_iot2023_passthrough():
    """CIC columns map directly; values should be preserved."""
    df = pd.DataFrame({
        "number": [7], "tot_size": [999],
        "iat": [3.14], "tcp": [0], "udp": [0], "icmp": [1],
        "label": [1], "type": ["probe"],
    })
    result = align_cic_iot2023(df)
    assert result["f_total_bytes"].tolist() == [999]
    assert result["f_duration"].tolist() == [3.14]
    assert result["f_is_icmp"].tolist() == [1]


def test_aligners_dict():
    assert "ton_iot" in ALIGNERS
    assert "cic_iot2023" in ALIGNERS
    assert ALIGNERS["ton_iot"] is align_ton_iot
    assert ALIGNERS["cic_iot2023"] is align_cic_iot2023

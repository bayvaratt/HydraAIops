from hydra.data.schema import run_smoke_checks


def test_schema_smoke(tmp_path):
    from tests.conftest import make_sample_df

    df = make_sample_df(n=50)
    out_dir = tmp_path / "schema"
    out_dir.mkdir()

    result = run_smoke_checks(
        df,
        dataset_name="unit",
        label_col="label",
        type_col="type",
        out_dir=str(out_dir),
        high_cardinality_threshold=5,
        high_cardinality_ratio=0.1,
    )

    assert (out_dir / "missingness.json").exists()
    assert (out_dir / "high_cardinality_columns.json").exists()
    assert (out_dir / "dataset_summary.json").exists()
    assert "summary" in result

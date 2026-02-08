from hydra.data.preprocess import TabularPreprocessor


def test_preprocess_behaviour_only_drops_identifiers():
    from tests.conftest import make_sample_df

    df = make_sample_df(n=100)
    pre = TabularPreprocessor(
        feature_regime="behaviour_only",
        label_col="label",
        type_col="type",
        high_cardinality_threshold=5,
        high_cardinality_ratio=0.1,
        categorical_top_k=10,
        enable_port_bucketing=True,
        port_top_n=5,
    )

    X_train = pre.fit_transform(df)
    assert "src_ip" in pre.dropped_columns
    assert "dst_ip" in pre.dropped_columns
    assert X_train.shape[0] == len(df)

    X_test = pre.transform(df.sample(n=10, random_state=1))
    assert X_test.shape[1] == X_train.shape[1]

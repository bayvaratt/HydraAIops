import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hydra.data.preprocess import TabularPreprocessor
from hydra.data.split import split_tabular


def _to_dense(mat):
    return mat.toarray() if hasattr(mat, "toarray") else mat


def test_train_tabular_models():
    from tests.conftest import make_sample_df

    df = make_sample_df(n=200)
    train_df, val_df, test_df = split_tabular(df, label_col="label", seed=42)

    pre = TabularPreprocessor(
        feature_regime="behaviour_only",
        label_col="label",
        type_col="type",
        high_cardinality_threshold=5,
        high_cardinality_ratio=0.1,
        categorical_top_k=10,
    )

    X_train = _to_dense(pre.fit_transform(train_df))
    X_test = _to_dense(pre.transform(test_df))
    y_train = train_df["label"].astype(int).to_numpy()

    models = [
        LogisticRegression(max_iter=2000, solver="lbfgs"),
        RandomForestClassifier(n_estimators=50, random_state=0),
    ]

    for model in models:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        assert proba.shape[0] == X_test.shape[0]

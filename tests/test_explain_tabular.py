import pytest
from sklearn.ensemble import RandomForestClassifier

from hydra.data.preprocess import TabularPreprocessor
from hydra.data.split import split_tabular
from hydra.explain.shap_tabular import explain_tree_shap
from hydra.eval.explainability import coverage


def test_tree_shap_explainability():
    pytest.importorskip("shap")
    from tests.conftest import make_sample_df

    df = make_sample_df(n=200)
    train_df, _, test_df = split_tabular(df, label_col="label", seed=42)

    pre = TabularPreprocessor(
        feature_regime="behaviour_only",
        label_col="label",
        type_col="type",
        high_cardinality_threshold=5,
        high_cardinality_ratio=0.1,
        categorical_top_k=10,
    )

    X_train = pre.fit_transform(train_df)
    X_test = pre.transform(test_df)
    y_train = train_df["label"].astype(int).to_numpy()

    model = RandomForestClassifier(n_estimators=50, random_state=0)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]

    ids = list(range(min(10, X_test.shape[0])))
    X_test_dense = X_test.toarray() if hasattr(X_test, "toarray") else X_test
    records = explain_tree_shap(
        model,
        X_test_dense,
        pre.feature_names,
        ids,
        scores,
        model_name="random_forest",
        feature_regime="behaviour_only",
        dataset_name="unit",
        seed=0,
        top_k=5,
    )

    assert len(records) == len(ids)
    cov = coverage(records, ids)
    assert cov > 0.95
    assert len(records[0].top_features) == 5

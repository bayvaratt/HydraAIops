"""Smoke tests for analysis modules — verify they import and have expected interfaces."""
import pytest


def test_aggregate_runs_importable():
    from hydra.analysis import aggregate_runs
    assert hasattr(aggregate_runs, 'main')


def test_make_report_plots_importable():
    from hydra.analysis import make_report_plots
    assert hasattr(make_report_plots, 'main')


def test_make_shap_plots_importable():
    from hydra.analysis import make_shap_plots
    assert hasattr(make_shap_plots, 'main')


def test_xai_diagnostics_importable():
    from hydra.analysis import xai_diagnostics
    assert hasattr(xai_diagnostics, 'run')


def test_xai_eval_report_importable():
    from hydra.analysis import xai_eval_report
    assert hasattr(xai_eval_report, 'run')


def test_accuracy_xai_tradeoff_importable():
    from hydra.analysis import accuracy_xai_tradeoff
    assert hasattr(accuracy_xai_tradeoff, 'run')


def test_run_experiments_importable():
    from hydra.experiments import run_experiments
    assert callable(getattr(run_experiments, 'main', None)) or hasattr(run_experiments, 'run')

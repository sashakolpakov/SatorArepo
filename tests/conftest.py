"""Shared fixtures and marker registration for arepo tests."""

import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast tests with no network or model dependencies")
    config.addinivalue_line("markers", "slow: integration tests that load transformer models")
    config.addinivalue_line("markers", "reproducibility: full benchmark pipeline tests")


@pytest.fixture
def sample_embedding():
    """Synthetic embedding tensor (1-D numpy array) for unit tests."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(1000).astype(np.float32)


@pytest.fixture
def benford_distribution():
    """Exact Benford PMF for digits 1-9."""
    digits = np.arange(1, 10)
    return np.log10(1 + 1.0 / digits)


@pytest.fixture
def sample_texts():
    """Load bundled sample data (12 human + 12 AI texts)."""
    from arepo.download import load_sample_texts
    return load_sample_texts()


@pytest.fixture
def sample_4d_data():
    """Synthetic 4D feature matrix + labels for engine tests."""
    rng = np.random.default_rng(42)
    n = 40
    human = rng.normal(loc=[0.3, 0.3, 0.2, 0.2], scale=0.05, size=(n, 4))
    ai = rng.normal(loc=[0.1, 0.1, 0.4, 0.4], scale=0.05, size=(n, 4))
    X = np.vstack([human, ai])
    y = np.array([0] * n + [1] * n)
    return X, y

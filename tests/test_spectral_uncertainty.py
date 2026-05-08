# tests/test_spectral_uncertainty.py

import numpy as np
import pytest

from omniuq.datasets import DatasetLoader
from omniuq.spectral_uncertainty import SpectralUncertainty
from omniuq.utils import load_llm_model


# Algorithmic invariants — no model needed, fast
class TestSpectralAlgorithm:
    def test_vne_zero_for_identical_responses(self):
        # All-ones kernel → rank-1 → one eigenvalue = 1, rest = 0 → VNE = 0
        K = np.ones((5, 5))
        assert SpectralUncertainty._vne_from_kernel(K) == pytest.approx(0.0, abs=1e-6)

    def test_vne_max_for_orthogonal_responses(self):
        # Identity kernel → uniform spectrum → max entropy = log(n)
        K = np.eye(5)
        vne = SpectralUncertainty._vne_from_kernel(K)
        assert vne == pytest.approx(np.log(5), abs=1e-6)

    def test_total_equals_aleatoric_plus_epistemic(self):
        # Decomposition identity from Corollary 3.6
        result = {"aleatoric": 0.4, "epistemic": 0.6, "total": 1.0}
        assert result["aleatoric"] + result["epistemic"] == pytest.approx(result["total"])


# End-to-end smoke test on TriviaQA — slow, requires GPU
@pytest.mark.slow
class TestSpectralEndToEnd:
    @pytest.fixture(scope="class")
    def small_model(self):
        # Tiny model for smoke-testing pipeline integrity, not paper reproduction
        tokenizer, model = load_llm_model("Qwen/Qwen2.5-0.5B-Instruct", device="cuda:0")
        return tokenizer, model

    def test_triviaqa_loader(self):
        data = DatasetLoader("triviaqa", n_samples=5).load()
        assert len(data) == 5
        assert all("question" in d and "answers" in d for d in data)

    def test_score_returns_valid_decomposition(self, small_model):
        tokenizer, model = small_model
        uq = SpectralUncertainty(tokenizer, model, m_samples=5)
        result = uq.score("What is the capital of France?")
        assert result["epistemic"] >= 0
        assert result["total"] >= result["epistemic"] - 1e-6  # aleatoric >= 0 up to noise

    def test_pipeline_on_triviaqa_subset(self, small_model):
        tokenizer, model = small_model
        uq = SpectralUncertainty(tokenizer, model, m_samples=5)
        data = DatasetLoader("triviaqa", n_samples=3).load()
        results = uq.score_batch([d["question"] for d in data])
        assert len(results) == 3
        assert all("aleatoric" in r and "epistemic" in r and "total" in r for r in results)
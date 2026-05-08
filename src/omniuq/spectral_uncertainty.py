# src/omniuq/spectral_uncertainty.py

from __future__ import annotations

import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import rbf_kernel
from tqdm.auto import tqdm

from .utils import OpenAIProvider, parametric_answer

# Defaults from Walha et al. 2025 (arXiv:2509.22272)
DEFAULT_EMBEDDER = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_M_SAMPLES = 10
DEFAULT_GAMMA = 1.0
EPS = 1e-12


CLARIFICATION_PROMPT = """Analyze the given question for ambiguities. If ambiguous, provide multiple clarifications that resolve the ambiguity. Each clarification must be a fully-specified question.

Rules:
- At most 10 clarifications.
- If unambiguous, output one paraphrase.
- Do not include answers.

Output format:
---Clarifications:
-1 [first clarification]
-2 [second clarification]
...

Question: {question}
"""

JUDGE_PROMPT = """Compare a model answer to a list of acceptable ground truth answers. The model answer is correct if it is semantically equivalent to ANY one of them, even if not lexically identical. Capitalization and minor wording do not matter. Output only "yes" or "no".

Question: {question}
Acceptable answers: {gold}
Model answer: {pred}
"""


class SpectralUncertainty:
    """Spectral Uncertainty (Walha et al. 2025).

    Decomposes predictive uncertainty into aleatoric and epistemic
    components via von Neumann entropy of kernel covariance operators
    over n clarifications x m sampled answers.
    """

    def __init__(
        self,
        target_tokenizer,
        target_model,
        clarifier=None,  # OpenAIProvider OR (tokenizer, hf_model) tuple
        judge=None,  # same options as clarifier
        max_clarifications: int = 10,
        embedder_name: str = DEFAULT_EMBEDDER,
        m_samples: int = DEFAULT_M_SAMPLES,
        temperature: float = DEFAULT_TEMPERATURE,
        gamma: float = DEFAULT_GAMMA,
        max_new_tokens: int = 64,
    ):
        self.tokenizer = target_tokenizer
        self.model = target_model
        self.clarifier = clarifier
        self.judge_provider = judge
        self.max_clarifications = max_clarifications
        self.embedder_name = embedder_name
        self.target_model_name = getattr(
            target_model.config, "_name_or_path", "unknown"
        )
        self.m = m_samples
        self.temperature = temperature
        self.gamma = gamma
        self.max_new_tokens = max_new_tokens
        self.embedder = SentenceTransformer(
            embedder_name, device=str(self.model.device)
        )

    # --- Backend-agnostic chat ---

    def _chat(self, provider, prompt: str, max_tokens: int = 256) -> str:
        # Two cases: an OpenAIProvider, or a (tokenizer, hf_model) tuple for a local HF chat model.
        if provider is None:
            return ""
        if isinstance(provider, tuple):
            tok, model = provider
            return parametric_answer(tok, model, prompt, max_new_tokens=max_tokens)
        if isinstance(provider, OpenAIProvider):
            return provider.chat(prompt, max_tokens=max_tokens)
        raise TypeError(
            f"Unsupported provider type: {type(provider).__name__}. "
            "Expected OpenAIProvider or (tokenizer, model) tuple."
        )

    # --- Clarification ---

    def _clarify(self, question: str) -> list[str]:
        if self.clarifier is None:
            return [question]
        text = self._chat(
            self.clarifier,
            CLARIFICATION_PROMPT.format(question=question),
            max_tokens=512,
        )
        matches = [m.strip() for m in re.findall(r"-\d+\s+(.+)", text) if m.strip()]
        return matches[: self.max_clarifications] if matches else [question]

    # --- Judging ---

    def judge(self, question: str, prediction: str, gold: list[str]) -> int:
        if self.judge_provider is None:
            raise RuntimeError("No judge provider was passed to SpectralUncertainty.")
        gold_str = "\n".join(f"- {g}" for g in gold)
        text = self._chat(
            self.judge_provider,
            JUDGE_PROMPT.format(question=question, gold=gold_str, pred=prediction),
            max_tokens=5,
        )
        return int("yes" in text.strip().lower())

    # --- Sampling ---

    @torch.inference_mode()
    def _sample_answers(self, prompt: str) -> list[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            num_return_sequences=self.m,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = outputs[:, inputs["input_ids"].shape[1] :]
        return [
            self.tokenizer.decode(g, skip_special_tokens=True)
            .strip()
            .split("\n")[0]
            .strip()
            for g in gen
        ]

    # --- Spectral math ---

    @staticmethod
    def _vne_from_kernel(K: np.ndarray) -> float:
        # von Neumann entropy = Shannon entropy of eigenvalues of K/n (Bach 2022)
        n = K.shape[0]
        eigvals = np.linalg.eigvalsh(K / n)
        eigvals = np.clip(eigvals, EPS, None)
        return float(-np.sum(eigvals * np.log(eigvals)))

    def score(self, question: str, verbose: bool = False) -> dict:
        clarifications = self._clarify(question)

        all_embeddings = []
        per_clarif_embeds = []
        all_answers = []

        for clar in clarifications:
            prompt = f"Question: {clar}\nAnswer:"
            answers = self._sample_answers(prompt)
            embeds = self.embedder.encode(answers, normalize_embeddings=True)
            per_clarif_embeds.append(embeds)
            all_embeddings.append(embeds)
            all_answers.append(answers)

        all_embeddings = np.vstack(all_embeddings)

        inner_vnes = [
            self._vne_from_kernel(rbf_kernel(E, gamma=self.gamma))
            for E in per_clarif_embeds
        ]
        epistemic = float(np.mean(inner_vnes))

        K_out = rbf_kernel(all_embeddings, gamma=self.gamma)
        total = self._vne_from_kernel(K_out)
        aleatoric = total - epistemic

        result = {
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total": total,
            "n_clarifications": len(clarifications),
            "m_samples": self.m,
        }
        if verbose:
            result["answers"] = all_answers
            result["clarifications"] = clarifications
            result["models"] = {
                "target": self.target_model_name,
                "embedder": self.embedder_name,
                "clarifier": self._provider_name(self.clarifier),
                "judge": self._provider_name(self.judge_provider),
            }
        return result

    @staticmethod
    def _provider_name(provider) -> str | None:
        if provider is None:
            return None
        if isinstance(provider, tuple):
            _, m = provider
            return getattr(m.config, "_name_or_path", "unknown")
        if isinstance(provider, OpenAIProvider):
            return provider.model
        return "unknown"

    def score_batch(self, questions: list[str], verbose: bool = False) -> list[dict]:
        return [
            self.score(q, verbose=verbose) for q in tqdm(questions, desc="Spectral UQ")
        ]

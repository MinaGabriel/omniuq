# src/omniuq/spectral_uncertainty.py

from __future__ import annotations

import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import rbf_kernel
from tqdm.auto import tqdm
from .utils import OpenAIProvider, generate_answers, parametric_answer

# Defaults from Walha et al. 2025 (arXiv:2509.22272)
DEFAULT_EMBEDDER = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_M_SAMPLES = 10
DEFAULT_GAMMA = 1.0
EPS = 1e-12

# Rephrasing prompt matches reference repo TriviaQA zero_shot_clarification template.
# Produces 5 semantically equivalent rephrasings, parsed via ### Rephrasings: / #N markers.
CLARIFICATION_PROMPT = """\
**Objective**
In this task, you will receive a question. Your goal is to generate multiple versions of the question that convey the same meaning as the original one.

**Important Rules**
1. Ensure that each rephrasing of the question is distinct from the others.
2. Ensure that all rephrasings of the question are semantically equivalent to the original question.
3. Provide 5 different rephrasings of the question.

**Output Format**
Your output should follow this format:
### Rephrasings:
#1 [Your rephrased question]
#2 [Another rephrased question]
#3 [Yet another rephrased question]
#4 [A fourth rephrasing of the question]
#5 [A fifth rephrasing of the question]

**Task Input**
### Original Question:
{question}"""

# Answer template matches reference repo TriviaQA generate_answer.txt + format_query() wrapper.
ANSWER_TEMPLATE = """\
**Objective**
In the following, I will provide a question and you need to provide an answer to the question. Your answer has to be short and precise. Do not write extra text or explanation, just give the answer directly. If the question is unclear or you do not know the answer, do not answer with phrases like "I'm sorry.." or "The question is unclear". Instead, you need to give a random guess for the answer. Do not ask follow-up questions or indicate that you do not know the answer. You should always provide a short and precise answer; either the true answer if you know it or your random guess if you are unsure. It should not be recognizable in your output whether your answer is the true answer or the random guess.
Your output should follow the format specified below in the Output Format section.

**Output Format**
A: [Your short and precise answer or random guess to the question. Do not include any additional information.]

**Task**
Question:
Q: {question}"""

# Judge prompt matches reference repo TriviaQA correctness_judge.txt + format_correctness_judge_query() wrapper.
JUDGE_PROMPT = """\
**Objective**
In this task, you will receive a question. You will also receive a ground truth answer to the question and a model generated answer. Your goal is to compare the ground truth answer and the model generated answer in order to decide whether the model generated answer is correct or not.

**Important Rules**
1. The model generated answer is correct, when it is a valid answer to the question, and semantically equivalent to the ground truth answer. It does not necessarily need to overlap with the ground truth answer lexically.
2. If the model generated answer contains more information (more specific) or less information (less specific) than the ground truth answer, but still correctly answers the question, then you should consider it correct.
3. If you decide that the model generated answer is correct, say yes, otherwise say no.
4. Your output should only contain your decision (yes or no). It should not contain any other text, explanation or reasoning.

**Input**
### Question:
{question}

### Ground Truth Answer:
{gold}

### Model Generated Answer:
{pred}

Is the model generated answer correct? Answer with yes or no."""


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
        max_clarifications: int = 5,
        embedder_name: str = DEFAULT_EMBEDDER,
        m_samples: int = DEFAULT_M_SAMPLES,
        temperature: float = DEFAULT_TEMPERATURE,
        gamma: float = DEFAULT_GAMMA,
        max_new_tokens: int = 100,
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
        # Parse "### Rephrasings:\n#1 ...\n#2 ..." format from reference repo
        matches = re.findall(r"#\d+\s+(.*?)(?=(?:---|#|\Z))", text, re.DOTALL)
        matches = [m.strip() for m in matches if m.strip()]
        return matches[: self.max_clarifications] if matches else [question]

    # --- Judging ---

    def judge(self, question: str, prediction: str, gold: list[str]) -> int:
        if self.judge_provider is None:
            raise RuntimeError("No judge provider was passed to SpectralUncertainty.")
        gold_str = " / ".join(gold)
        text = self._chat(
            self.judge_provider,
            JUDGE_PROMPT.format(question=question, gold=gold_str, pred=prediction),
            max_tokens=5,
        )
        return int("yes" in text.strip().lower())

    # --- Sampling ---

    @torch.inference_mode()
    def _sample_answers(self, question: str) -> list[str]:
        prompt = ANSWER_TEMPLATE.format(question=question)
        return generate_answers(
            self.tokenizer,
            self.model,
            prompt,
            n=self.m,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_p=0.95,
            terse=False,
        )

    @torch.inference_mode()
    def greedy_answer(self, question: str, max_new_tokens: int = 100) -> str:
        prompt = ANSWER_TEMPLATE.format(question=question)
        return generate_answers(
            self.tokenizer,
            self.model,
            prompt,
            n=1,
            temperature=0.1,
            max_new_tokens=max_new_tokens,
            terse=False,
        )[0]

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
            answers = self._sample_answers(clar)
            # No normalize_embeddings — matches reference repo encoding_arguments
            embeds = self.embedder.encode(answers, convert_to_numpy=True)
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

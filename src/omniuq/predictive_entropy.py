# src/omniuq/predictive_entropy.py
#
# Predictive Entropy (PE) for free-form LLM answers.
#
# Estimator: Malinin & Gales (ICLR 2021).
# Protocol:  Kuhn, Gal & Farquhar (ICLR 2023), "Semantic Uncertainty", and their code
#            (github.com/lorenzkuhn/semantic_uncertainty).
#
#   log p(s | x) = sum of the token log-probs of answer s
#   PE(x)        = -(1/M) * sum over M sampled answers of log p(s | x)
#
# Higher PE = the model is less sure.
# Answers are sampled at T = 0.5, then scored by the normal model (T = 1), as in the paper.

from __future__ import annotations

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

# First line of the paper's TriviaQA prompt.
KUHN_PREAMBLE = "This is a bot that correctly answers questions. \n"

# The model may not start these words while answering (paper's banned tokens).
KUHN_BANNED_STRINGS = ("Question:", " Question:", "\n", "Answer:", " Answer:", "Q:")

# An answer is cut at the first of these strings (paper's cleaning step).
KUHN_CLEAN_STRINGS = (".", "\n", "Q:", "A:", "question:", "answer:", "Question:", "Answer:",
                      "Questions:", "questions:", "QUESTION:", "ANSWER:")


# ---------- Prompt and cleaning ----------

def build_prompt(question: str, demos: list[tuple[str, str]] | None = None,
                 preamble: str = KUHN_PREAMBLE) -> str:
    """Paper format: preamble + "Question: q Answer: a " per demo + "Question: {question} Answer:"."""
    shots = "".join(f"Question: {q} Answer: {a} " for q, a in (demos or []))
    return f"{preamble}{shots}Question: {question} Answer:"


def clean_generation(text: str, strings: tuple[str, ...] = KUHN_CLEAN_STRINGS) -> str:
    """Keep the text before the first cut string. No stripping, as in the paper's code."""
    for s in strings:
        if s in text:
            text = text.split(s)[0]
    return text


# ---------- The estimator ----------

def predictive_entropy(token_logprobs: list[list[float]]) -> float:
    """PE = minus the average answer log-probability. Empty answers are skipped."""
    seq = [float(np.sum(lp)) for lp in token_logprobs if len(lp) > 0]
    return -float(np.mean(seq)) if seq else float("nan")


# ---------- Generation and scoring ----------

class _StopOnStrings(StoppingCriteria):
    """Stop a sample as soon as it contains a cut string (only saves time)."""

    def __init__(self, tokenizer, prompt_len: int, strings: tuple[str, ...]):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.strings = strings

    def __call__(self, input_ids, scores, **kwargs):
        texts = self.tokenizer.batch_decode(input_ids[:, self.prompt_len:], skip_special_tokens=True)
        return torch.tensor([any(s in t for s in self.strings) for t in texts],
                            dtype=torch.bool, device=input_ids.device)


class PredictiveEntropy:
    """Predictive entropy with the Kuhn et al. (2023) settings.

    Steps for one prompt:
      1. sample M answers at temperature T
      2. clean each answer
      3. score each answer's tokens with the model
      4. PE = -mean(answer log-probs)
      5. beam search for the most likely answer (used to grade correctness)

    stop_at_eos: False = paper (stop only at "."). True = also stop at the model's end-of-text token.
    """

    def __init__(self, model, tokenizer, num_samples: int = 10, temperature: float = 0.5,
                 top_k: int = 50, top_p: float = 1.0, max_new_tokens: int = 256, num_beams: int = 5,
                 stop_token: str = ".", stop_at_eos: bool = False,
                 banned_strings: tuple[str, ...] = KUHN_BANNED_STRINGS,
                 clean_strings: tuple[str, ...] = KUHN_CLEAN_STRINGS):
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples        # M (paper: 10)
        self.temperature = temperature        # paper: 0.5
        self.top_k = top_k                    # HuggingFace default the paper ran with
        self.top_p = top_p                    # paper: 1.0
        self.max_new_tokens = max_new_tokens  # paper: 256
        self.num_beams = num_beams            # paper: 5
        self.clean_strings = tuple(clean_strings)

        # Generation stops at this token (paper: ".").
        stop = tokenizer(stop_token, add_special_tokens=False).input_ids
        if len(stop) != 1:
            raise ValueError(f"stop_token {stop_token!r} is not a single token for this tokenizer.")
        self.stop_ids = [stop[0]]
        if stop_at_eos and tokenizer.eos_token_id is not None and tokenizer.eos_token_id != stop[0]:
            self.stop_ids.append(tokenizer.eos_token_id)

        # Ban the first token of each banned string (no duplicates).
        first_ids = dict.fromkeys(tokenizer(s, add_special_tokens=False).input_ids[0] for s in banned_strings)
        self.bad_words_ids = [[t] for t in first_ids]

        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def encode(self, prompt: str) -> torch.Tensor:
        """Prompt -> token ids on the model's device (BOS added, as in the paper)."""
        return self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

    @torch.inference_mode()
    def sample(self, prompt_ids: torch.Tensor, seed: int | None = None) -> list[str]:
        """M sampled answers, cleaned."""
        if seed is not None:
            torch.manual_seed(seed)
        p_len = prompt_ids.shape[1]
        out = self.model.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            do_sample=True,
            num_beams=1,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            num_return_sequences=self.num_samples,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.stop_ids,
            pad_token_id=self.pad_id,
            bad_words_ids=self.bad_words_ids,
            stopping_criteria=StoppingCriteriaList([_StopOnStrings(self.tokenizer, p_len, self.clean_strings)]),
        )
        texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in out[:, p_len:]]
        return [clean_generation(t, self.clean_strings) for t in texts]

    @torch.inference_mode()
    def most_likely(self, prompt_ids: torch.Tensor) -> str:
        """Most likely answer by beam search. Returned raw (not cleaned): the paper grades it this way."""
        p_len = prompt_ids.shape[1]
        out = self.model.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            do_sample=False,
            num_beams=self.num_beams,
            num_return_sequences=1,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.stop_ids,
            pad_token_id=self.pad_id,
            bad_words_ids=self.bad_words_ids,
        )
        return self.tokenizer.decode(out[0, p_len:], skip_special_tokens=True).lstrip()

    @torch.inference_mode()
    def token_logprobs(self, prompt_ids: torch.Tensor, answers: list[str]) -> list[list[float]]:
        """Log-prob of every token of every answer, given the prompt. One batched forward pass."""
        dev = prompt_ids.device
        p_len = prompt_ids.shape[1]
        ans_ids = [self.tokenizer(a, add_special_tokens=False).input_ids for a in answers]
        max_len = max((len(a) for a in ans_ids), default=0)
        if max_len == 0:
            return [[] for _ in answers]

        # Build one batch: prompt + answer, padded on the right.
        ids = torch.full((len(ans_ids), p_len + max_len), self.pad_id, dtype=torch.long, device=dev)
        mask = torch.zeros_like(ids)
        ids[:, :p_len] = prompt_ids[0]
        mask[:, :p_len] = 1
        for j, a in enumerate(ans_ids):
            if a:
                ids[j, p_len:p_len + len(a)] = torch.tensor(a, dtype=torch.long, device=dev)
                mask[j, p_len:p_len + len(a)] = 1

        # The logit at position t predicts token t + 1, so answers start at p_len - 1.
        logits = self.model(input_ids=ids, attention_mask=mask).logits[:, p_len - 1:p_len - 1 + max_len]
        logp = torch.log_softmax(logits.float(), dim=-1)

        out = []
        for j, a in enumerate(ans_ids):
            if not a:
                out.append([])
                continue
            targets = torch.tensor(a, dtype=torch.long, device=dev)
            scores = logp[j, :len(a)].gather(1, targets.unsqueeze(1)).squeeze(1)
            out.append(scores.cpu().tolist())
        return out

    def score(self, prompt: str, seed: int | None = None, most_likely: bool = True) -> dict:
        """PE for one prompt, with the samples and the most likely answer."""
        ids = self.encode(prompt)
        samples = self.sample(ids, seed)
        lps = self.token_logprobs(ids, samples)
        out = {
            "pe": predictive_entropy(lps),
            "samples": samples,
            "seq_logprobs": [float(np.sum(lp)) for lp in lps],
        }
        if most_likely:
            raw = self.most_likely(ids)
            out["answer_raw"] = raw                                              # used for grading
            out["answer"] = clean_generation(raw, self.clean_strings).strip()   # for reading
        return out


def run_predictive_entropy(model, tokenizer, question: str, demos: list[tuple[str, str]] | None = None,
                           num_samples: int = 10, temperature: float = 0.5, stop_at_eos: bool = False,
                           seed: int | None = None) -> dict:
    """Score one question with the paper's settings."""
    pe = PredictiveEntropy(model, tokenizer, num_samples=num_samples, temperature=temperature,
                           stop_at_eos=stop_at_eos)
    return {"question": question, **pe.score(build_prompt(question, demos), seed=seed)}


# ---------- Grading and evaluation (as in the paper) ----------

_ROUGE_SCORER = None


def rouge_l(prediction: str, reference: str) -> float:
    """Rouge-L F-measure, no stemming."""
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        try:
            from rouge_score import rouge_scorer
        except ImportError as e:
            raise ImportError("Install with: pip install rouge-score") from e
        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return float(_ROUGE_SCORER.score(reference, prediction)["rougeL"].fmeasure)


def rouge_correct(prediction: str, reference: str, threshold: float = 0.3) -> bool:
    """Paper's correctness label: Rouge-L > 0.3."""
    return rouge_l(prediction, reference) > threshold


def auroc(uncertainty, correct) -> float:
    """How well uncertainty picks out wrong answers. Same as the paper: roc_auc_score(1 - correct, PE)."""
    from sklearn.metrics import roc_auc_score

    u, y = np.asarray(uncertainty, float), np.asarray(correct, int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(1 - y, u))


def bootstrap_auroc(uncertainty, correct, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap confidence interval for auroc()."""
    u, y = np.asarray(uncertainty, float), np.asarray(correct, int)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(u), len(u))
        if y[idx].min() != y[idx].max():
            stats.append(auroc(u[idx], y[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)
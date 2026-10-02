# src/omniuq/ptrue_kadavath.py
#
# P(True) self-evaluation from Kadavath et al. (2022), "Language Models (Mostly) Know What They Know".
# Method only: answer sampling, the P(True) readout and the predictive-entropy baseline.
# Data loading lives in omniuq.datasets; TriviaQA grading and few-shot sampling in omniuq.utils.

from __future__ import annotations

import math

import numpy as np
import torch

# The paper's six hand-written self-evaluation examples (Appendix A.5).
A5_DEMOS = [
    {"question": "Who was the third president of the United States?",
     "generated_texts": ["James Monroe", "Thomas Jefferson", "Jefferson", "Thomas Jefferson", "George Washington"],
     "prediction": "James Monroe", "correct": False},
    {"question": "Calculate 33 + 4",
     "generated_texts": ["37", "37", "40", "36", "37"],
     "prediction": "37", "correct": True},
    {"question": 'Fill in the blank in the sentence "I went to the grocery and then to the pharmacy. I was '
                 'disappointed that they didn\'t have any vegetarian sausage at the _____."',
     "generated_texts": ["grocery", "store", "grocery", "refrigerator", "grocery"],
     "prediction": "grocery", "correct": True},
    {"question": "Name a celebrated civil rights leader.",
     "generated_texts": ["Martin Luther King", "Ghandhi", "Martin Luther King", "Barack Obama", "Martin Luther King"],
     "prediction": "Martin Luther King", "correct": True},
    {"question": "Calculate 33 * 849",
     "generated_texts": ["28,347", "1,490", "27,488", "3,409", "34,561"],
     "prediction": "28347", "correct": False},
    {"question": 'Fill in the blank in the sentence "I shot the _____ and it went swish. We walked away the '
                 'winners of that battle!"',
     "generated_texts": ["gun", "bullet", "arrow", "basketball", "rifle"],
     "prediction": "gun", "correct": False},
]

# Default Question/Answer pairs that teach a base model the short-answer format (App. A.7 style).
# They are TriviaQA validation questions: when evaluating on TriviaQA, pass your own demos instead.
DEFAULT_GEN_DEMOS = [
    ("Anatomy. Where are the intercostal muscles situated?", "between the ribs"),
    ("What Greek mythological horse sprang from the blood of Medusa?", "Pegasus"),
    ("In the Bible who was David's father?", "Jesse"),
    ("What type of tower is the Muslim version of a steeple?", "Minaret"),
    ("Which band released the albums Sheer Heart Attack and A Day at the Races?", "Queen"),
]


def ptrue_block(question: str, ideas: list[str], answer: str, label: str | None = None,
                brainstorm: bool = True) -> str:
    """One self-evaluation block: Sec. 4.2 format if brainstorm=True, Sec. 4.1 format otherwise."""
    if brainstorm:
        s = (f"Question: {question}\nHere are some brainstormed ideas: " + "\n".join(ideas)
             + f"\nPossible Answer: {answer}\nIs the possible answer:\n (A) True\n (B) False\n"
             "The possible answer is:")
    else:
        s = (f"Question: {question}\nProposed Answer: {answer}\nIs the proposed answer:\n"
             " (A) True\n (B) False\nThe proposed answer is:")
    # Demonstrations carry their label; the evaluated block ends right before it.
    return s + (f" ({label})" if label else "")


class PTrue:
    """P(True) self-evaluation (Kadavath et al., 2022, Sec. 4).

    Reads the next-token probabilities of "A" (True) and "B" (False) after
    "The possible answer is: (" and returns P(True) = p(A) / (p(A) + p(B)).
    Use a pretrained (base) model with plain-text prompts, as in the paper.

    Args:
        demonstrations: few-shot blocks, each a dict with question, generated_texts,
            prediction and correct. A5_DEMOS gives the paper's fixed prompt; 20 labelled
            examples from the evaluation set give the paper's 20-shot setup.
        brainstorm: True shows the sampled answers (Sec. 4.2 "comparison");
            False judges the answer alone (Sec. 4.1 "basic").
    """

    def __init__(self, model, tokenizer, demonstrations: list[dict] | None = None, brainstorm: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.demonstrations = demonstrations or []
        self.brainstorm = brainstorm
        # The letter must be its own token right after " (", otherwise the readout position is wrong.
        self.open_ids = tokenizer(" (", add_special_tokens=False).input_ids
        self.a_id = tokenizer(" (A", add_special_tokens=False).input_ids[-1]
        self.b_id = tokenizer(" (B", add_special_tokens=False).input_ids[-1]
        if len(self.open_ids) != 1 or \
                tokenizer(" (A", add_special_tokens=False).input_ids != self.open_ids + [self.a_id] or \
                tokenizer(" (B", add_special_tokens=False).input_ids != self.open_ids + [self.b_id]:
            raise ValueError("This tokenizer does not split ' (A' / ' (B' into ' (' + letter; P(True) readout unsupported.")

    def build_prompt(self, sampled_gen_dict: dict, prediction: str) -> str:
        shots = [ptrue_block(d["question"], d["generated_texts"], d["prediction"],
                             "A" if d["correct"] else "B", self.brainstorm) for d in self.demonstrations]
        target = ptrue_block(sampled_gen_dict["question"], sampled_gen_dict["generated_texts"],
                             prediction, None, self.brainstorm)
        return "\n\n".join(shots + [target])

    @torch.inference_mode()
    def __call__(self, sampled_gen_dict: dict, prediction: str, context: str | None = None) -> dict:
        prompt = self.build_prompt(sampled_gen_dict, prediction)
        full = self.tokenizer(prompt + " (").input_ids
        if full != self.tokenizer(prompt).input_ids + self.open_ids:
            raise ValueError("' (' merged with the end of the prompt; the letter would be read at the wrong position.")
        ids = torch.tensor([full], device=self.model.device)

        # Hidden states at the last two positions: before " (" and before the letter.
        h = self.model.model(input_ids=ids).last_hidden_state[0, -2:]
        head = self.model.get_output_embeddings()
        logits = head(h).float()
        logp = torch.log_softmax(logits, dim=-1)

        # The A/B logits in float32 from two rows of the LM head (bf16 logits are rounded).
        # Falls back to the regular logits if the head is offloaded or quantized.
        if head.weight.device.type == "meta" or head.weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            la, lb = logits[1, self.a_id].item(), logits[1, self.b_id].item()
        else:
            w = head.weight[[self.a_id, self.b_id]].to(h.device).float()
            la, lb = (w @ h[-1].float()).tolist()
        p_true = 1.0 / (1.0 + math.exp(lb - la))

        # Unnormalized probabilities of " (A" and " (B"; mass_AB should be close to 1.
        log_open = logp[0, self.open_ids[0]]
        p_a = (log_open + logp[1, self.a_id]).exp().item()
        p_b = (log_open + logp[1, self.b_id]).exp().item()
        return {"confidence": p_true, "uncertainty": 1.0 - p_true,
                "p_A_raw": p_a, "p_B_raw": p_b, "mass_AB": p_a + p_b}


@torch.inference_mode()
def sample_answers(model, tokenizer, question: str, demos: list[tuple[str, str]] | None = None,
                   num_samples: int = 5, max_new_tokens: int = 32) -> dict:
    """Sample answers at T=1 (no top-k/top-p) from a few-shot Question/Answer prompt.

    Returns {"question", "generated_texts", "logprobs"}. Log-probs come from the raw logits of the
    sampled tokens (output_scores would be temperature-scaled and truncated) and include the newline.
    """
    demos = DEFAULT_GEN_DEMOS if demos is None else demos
    prompt = "\n\n".join([f"Question: {q}\nAnswer: {a}" for q, a in demos] + [f"Question: {question}\nAnswer:"])
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=1.0, top_k=0,
                         top_p=1.0, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                         num_return_sequences=num_samples, stop_strings=["\n"], tokenizer=tokenizer,
                         pad_token_id=pad, output_logits=True, return_dict_in_generate=True)
    seqs = out.sequences[:, ids.shape[1]:]
    lps = torch.log_softmax(torch.stack(out.logits, 1).float(), -1).gather(-1, seqs[..., None])[..., 0]

    texts, logprobs = [], []
    for s, lp in zip(seqs.tolist(), lps.tolist()):
        # Keep tokens up to and including the first newline (or EOS); drop padding after it.
        n = next((i + 1 for i, t in enumerate(s) if t == tokenizer.eos_token_id or "\n" in tokenizer.decode([t])), len(s))
        texts.append(tokenizer.decode(s[:n], skip_special_tokens=True).split("\n")[0].strip())
        logprobs.append(lp[:n])
    return {"question": question, "generated_texts": texts, "logprobs": logprobs}


def predictive_entropy(logprobs: list[list[float]]) -> float:
    """PE = -(1/M) sum_m log p(s_m | x) over M samples at T=1 (App. B.2 estimator). Higher = less certain."""
    return -float(np.mean([sum(lp) for lp in logprobs]))


def run_kadavath(model, tokenizer, question: str, num_samples: int = 5, demonstrations: list[dict] | None = None,
                 gen_demos: list[tuple[str, str]] | None = None, brainstorm: bool = True) -> dict:
    """Score one question: sample answers, take the first as the prediction, return P(True) and PE.

    Defaults to the paper's fixed A.5 prompt with brainstormed ideas ("Comparison Examples (Prompt)").
    """
    g = sample_answers(model, tokenizer, question, gen_demos, num_samples)
    prediction = g["generated_texts"][0]
    r = PTrue(model, tokenizer, A5_DEMOS if demonstrations is None else demonstrations, brainstorm)(g, prediction)
    return {"question": question, "answer": prediction, "samples": g["generated_texts"],
            "p_true": r["confidence"], "mass_AB": r["mass_AB"], "pe": predictive_entropy(g["logprobs"])}



def auroc(scores, labels) -> float:
    """Rank-based AUROC with tie handling (same as sklearn). NaN if only one class is present.

    Higher scores should mean "more likely correct"; negate uncertainty scores such as entropy.
    """
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks, sorted_s, i = np.empty(len(s)), s[order], 0
    while i < len(s):
        # Tied scores share their average rank
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def brier_score(probs, labels) -> float:
    """Mean squared error between probabilities and 0/1 outcomes (App. A.3). Lower is better."""
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    return float(np.mean((p - y) ** 2))


def calibration_bins(probs, labels, n_bins: int = 10) -> list[tuple[float, float]]:
    """(mean probability, fraction correct) per bin, with equal numbers of predictions per bin (App. A.1)."""
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    bins = [b for b in np.array_split(np.argsort(p), n_bins) if len(b)]
    return [(float(p[b].mean()), float(y[b].mean())) for b in bins]


def ece_equal_mass(probs, labels, n_bins: int = 10) -> float:
    """Unweighted mean |fraction correct - mean probability| over equal-count bins (App. A.2).

    Differs from omniuq.utils.expected_calibration_error (equal-width, weighted bins).
    """
    return float(np.mean([abs(f - m) for m, f in calibration_bins(probs, labels, n_bins)]))


def selective_accuracy(probs, labels, threshold: float = 0.5) -> tuple[float, float]:
    """Accuracy among predictions with probability > threshold, and the fraction kept (coverage)."""
    p, y = np.asarray(probs, float), np.asarray(labels, float)
    keep = p > threshold
    acc = float(y[keep].mean()) if keep.any() else float("nan")
    return acc, float(keep.mean())


def confidence_metrics(probs, labels, threshold: float = 0.5, n_bins: int = 10) -> dict:
    """All metrics for a probability-valued confidence score (e.g. P(True))."""
    acc, coverage = selective_accuracy(probs, labels, threshold)
    return {"auroc": auroc(probs, labels), "brier": brier_score(probs, labels),
            "ece": ece_equal_mass(probs, labels, n_bins),
            f"acc_conf>{threshold}": acc, "coverage": coverage}
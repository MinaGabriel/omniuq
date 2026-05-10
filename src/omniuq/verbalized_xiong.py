"""
verbalized_xiong.py
-------------------
Full implementation of Xiong et al. (2024)'s verbalized confidence framework
for open-ended short-phrase factual QA.

Reference:
  Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Hooi, B.
  "Can LLMs Express Their Uncertainty? An Empirical Evaluation of
   Confidence Elicitation in LLMs." ICLR 2024.
  https://github.com/MiaoXiong2320/llm-uncertainty
  (Top-K prompting from Tian et al. 2023, arXiv:2305.14975)

Framework: Prompting x Sampling x Aggregation
  - Prompting:    Vanilla | CoT | TopK | MultiStep
  - Sampling:     M=1 (single greedy) or M>1 Self-Random (temperature>0)
  - Aggregation:  Consistency | AvgConf | PairRank

Adaptation note:
  Xiong et al.'s `answer_type` slot is "option letter" / "number".
  We adapt to "entity name or short phrase" for open-ended factual QA.
  This is a domain adaptation, not a method change.

Usage examples:
  # Strongest single config (paper's headline baseline):
  result = run_xiong(model, tok, q, device,
                     prompting="cot", n_samples=5, aggregation="avg_conf")

  # Vanilla baseline (the weak one, to demonstrate collapse):
  result = run_xiong(model, tok, q, device,
                     prompting="vanilla", n_samples=1, aggregation="avg_conf")

  # Run all configs at once (returns dict of results):
  results = run_xiong_grid(model, tok, q, device, n_samples=5)

  Each result dict now includes:
    - total_generation_tokens : total number of new tokens generated (sum over samples)
    - execution_time_seconds  : wall‑clock time for the generation step
"""

import re
import time
import torch
from collections import Counter
from typing import List, Tuple, Optional, Dict

# =============================================================================
# 1. PROMPTS  (verbatim from Xiong et al. 2024 / Tian et al. 2023)
# =============================================================================
ANSWER_TYPE = "entity name or short phrase"

VANILLA_PROMPT = (
    "Read the question, provide your answer and your confidence in this "
    "answer. Note: The confidence indicates how likely you think your "
    "answer is true.\n"
    "Use the following format to answer:\n"
    "```Answer and Confidence (0-100): [ONLY the {answer_type}; not a "
    "complete sentence], [Your confidence level, please only include the "
    "numerical number in the range of 0-100]%```\n"
    "Only the answer and confidence, don't give me the explanation.\n"
    "Question: {question}"
)

COT_PROMPT = (
    "Read the question, analyze step by step, provide your answer and "
    "your confidence in this answer. Note: The confidence indicates how "
    "likely you think your answer is true.\n"
    "Use the following format to answer:\n"
    "```Explanation: [insert step-by-step analysis here]\n"
    "Answer and Confidence (0-100): [ONLY the {answer_type}; not a complete "
    "sentence], [Your confidence level, please only include the numerical "
    "number in the range of 0-100]%\n```\n"
    "Only give me the reply according to this format, don't give me any "
    "other words.\n"
    "Question: {question}"
)

# Top-K (Tian et al. 2023, used by Xiong as a prompting variant)
TOPK_PROMPT = (
    "Provide your {k} best guesses and the probability that each is correct "
    "(0.0 to 1.0) for the following question. Give ONLY the guesses and "
    "probabilities, no other words or explanation. For example:\n\n"
    "G1: <first most likely guess, as short as possible; not a complete "
    "sentence, just the guess!>\n"
    "P1: <the probability between 0.0 and 1.0 that G1 is correct, without "
    "any extra commentary whatsoever; just the probability!>\n"
    "...\n"
    "G{k}: <{k}-th most likely guess, as short as possible; not a complete "
    "sentence, just the guess!>\n"
    "P{k}: <the probability between 0.0 and 1.0 that G{k} is correct, "
    "without any extra commentary whatsoever; just the probability!>\n\n"
    "The question is: {question}"
)

# Multi-Step (Xiong et al. 2024) — break into K reasoning steps each with conf
MULTI_STEP_PROMPT = (
    "Read the question, break down the problem into K steps, think step by "
    "step, give your confidence in each step, and then derive your final "
    "answer and your confidence in this answer. Note: The confidence "
    "indicates how likely you think your answer is true.\n"
    "Use the following format to answer:\n"
    "```Step 1: [Your reasoning], Confidence: [ONLY the confidence value "
    "that this step is correct]%\n"
    "...\n"
    "Step K: [Your reasoning], Confidence: [ONLY the confidence value that "
    "this step is correct]%\n"
    "Final Answer and Overall Confidence (0-100): [ONLY the {answer_type}; "
    "not a complete sentence], [Your confidence level, please only include "
    "the numerical number in the range of 0-100]%```\n"
    "Question: {question}"
)


# =============================================================================
# 2. PARSERS
# =============================================================================
# Vanilla / CoT / Multi-Step all end with the same line:
#   "Answer and Confidence (0-100): <answer>, <conf>%"
#   "Final Answer and Overall Confidence (0-100): <answer>, <conf>%"
_ANS_CONF_RE = re.compile(
    r"(?:Final\s+Answer\s+and\s+Overall\s+Confidence|Answer\s+and\s+Confidence)"
    r"[^:]*:\s*(?P<answer>.+?)\s*,\s*(?P<conf>\d{1,3}(?:\.\d+)?)\s*%?",
    re.IGNORECASE | re.DOTALL,
)


def _parse_answer_conf(raw: str) -> Tuple[Optional[str], Optional[float]]:
    """For Vanilla / CoT / Multi-Step. Returns (answer, conf in [0,1])."""
    m = _ANS_CONF_RE.search(raw)
    if not m:
        return None, None
    ans = m.group("answer").strip().strip(".,;:\"'`")
    ans = ans.split("\n")[0].strip()  # first line only
    try:
        conf = float(m.group("conf"))
    except ValueError:
        return (ans or None), None
    if conf > 1.0:  # "95" form
        conf = conf / 100.0
    conf = max(0.0, min(1.0, conf))
    return (ans or None), conf


# Top-K parser:  G1: ... \n P1: 0.92 \n G2: ... \n P2: 0.04 ...
_GUESS_RE = re.compile(r"G\s*(\d+)\s*:\s*(.+?)(?=\n|$)", re.IGNORECASE)
_PROB_RE = re.compile(r"P\s*(\d+)\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _parse_topk(raw: str) -> List[Tuple[str, float]]:
    """Returns list of (guess_str, probability) sorted by index."""
    guesses = {int(i): g.strip().strip(".,;:\"'`") for i, g in _GUESS_RE.findall(raw)}
    probs = {int(i): float(p) for i, p in _PROB_RE.findall(raw)}
    pairs = []
    for i in sorted(guesses):
        if i in probs and guesses[i]:
            p = probs[i]
            if p > 1.0:  # rare, but model sometimes outputs 95
                p = p / 100.0
            pairs.append((guesses[i], max(0.0, min(1.0, p))))
    return pairs


def _normalize(s: str) -> str:
    """Loose normalization for majority voting on entity answers."""
    return re.sub(r"\s+", " ", s.lower().strip().strip(".,;:\"'`"))


# =============================================================================
# 3. SAMPLING (one call -> n_samples raw strings + token counts)
# =============================================================================
@torch.no_grad()
def _sample_raw(
    model,
    tok,
    prompt: str,
    device,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> Tuple[List[str], List[int], float]:
    """
    Returns:
        texts: list of generated strings (decoded, skip_special_tokens)
        token_counts: list of number of newly generated tokens per sample
        elapsed_time: total time spent in generation (seconds)
    """
    inputs = tok(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[-1]
    do_sample = (n_samples > 1) or (temperature > 0)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_return_sequences=n_samples,
        pad_token_id=tok.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    start_time = time.perf_counter()
    out = model.generate(**inputs, **gen_kwargs)
    elapsed = time.perf_counter() - start_time

    # extract token counts (length of generated part only)
    token_counts = []
    texts = []
    for seq in out:
        gen_ids = seq[input_len:]
        token_counts.append(len(gen_ids))
        texts.append(tok.decode(gen_ids, skip_special_tokens=True))

    return texts, token_counts, elapsed


# =============================================================================
# 4. AGGREGATION
# =============================================================================
def _aggregate_consistency(parsed: List[Tuple[str, float]]) -> Dict:
    """Confidence = fraction of samples that agree with the majority answer.
    Ignores the verbalized number entirely (this is essentially AU)."""
    if not parsed:
        return {"answer": "", "confidence": 0.0, "majority_share": 0.0}
    norms = [_normalize(a) for a, _ in parsed]
    maj_norm, maj_count = Counter(norms).most_common(1)[0]
    share = maj_count / len(parsed)
    maj_ans = next(a for a, _ in parsed if _normalize(a) == maj_norm)
    return {"answer": maj_ans, "confidence": share, "majority_share": share}


def _aggregate_avg_conf(parsed: List[Tuple[str, float]]) -> Dict:
    """Avg-Conf: average of verbalized confidences over samples agreeing
    with majority answer, weighted by majority share. This is Xiong's
    strongest aggregation."""
    if not parsed:
        return {"answer": "", "confidence": 0.5, "majority_share": 0.0}
    norms = [_normalize(a) for a, _ in parsed]
    maj_norm, maj_count = Counter(norms).most_common(1)[0]
    share = maj_count / len(parsed)
    maj_ans = next(a for a, _ in parsed if _normalize(a) == maj_norm)
    agreeing = [c for a, c in parsed if _normalize(a) == maj_norm]
    avg = sum(agreeing) / len(agreeing)
    # Xiong's actual formula: share * avg_conf  (penalize disagreement)
    return {
        "answer": maj_ans,
        "confidence": float(share * avg),
        "majority_share": float(share),
        "raw_avg_conf": float(avg),  # for ablation
    }


def _aggregate_pair_rank(parsed: List[Tuple[str, float]]) -> Dict:
    """Pair-Rank approximation: contrast agreeing vs disagreeing confidences.
    Slightly more robust to noisy disagreers than Avg-Conf."""
    if not parsed:
        return {"answer": "", "confidence": 0.5, "majority_share": 0.0}
    norms = [_normalize(a) for a, _ in parsed]
    maj_norm, maj_count = Counter(norms).most_common(1)[0]
    share = maj_count / len(parsed)
    maj_ans = next(a for a, _ in parsed if _normalize(a) == maj_norm)
    agreeing = [c for a, c in parsed if _normalize(a) == maj_norm]
    disagreeing = [c for a, c in parsed if _normalize(a) != maj_norm]
    pos = sum(agreeing) / len(agreeing) if agreeing else 0.0
    neg = sum(disagreeing) / len(disagreeing) if disagreeing else 0.0
    score = pos - 0.5 * neg
    score = max(0.0, min(1.0, score))
    return {
        "answer": maj_ans,
        "confidence": float(score),
        "majority_share": float(share),
    }


_AGGREGATORS = {
    "consistency": _aggregate_consistency,
    "avg_conf": _aggregate_avg_conf,
    "pair_rank": _aggregate_pair_rank,
}


# =============================================================================
# 5. TOP-K HANDLING (different output format -> different aggregation)
# =============================================================================
def _topk_to_majority(samples_topk: List[List[Tuple[str, float]]]) -> Dict:
    """For Top-K, each sample is a *list* of (guess, prob).
    We take the top-1 guess from each sample and its associated prob,
    then aggregate via Avg-Conf-style logic."""
    parsed = []
    for sample in samples_topk:
        if sample:
            parsed.append(sample[0])  # top-1 (guess, prob)
    return _aggregate_avg_conf(parsed)


# =============================================================================
# 6. MAIN ENTRY POINTS (with timing & token counting)
# =============================================================================
@torch.no_grad()
def run_xiong(
    model,
    tok,
    question: str,
    device,
    prompting: str = "cot",
    n_samples: int = 5,
    aggregation: str = "avg_conf",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: Optional[int] = None,
    topk_k: int = 4,
) -> Dict:
    """Run one cell of the Xiong et al. (2024) framework.

    Args:
        prompting:    "vanilla" | "cot" | "topk" | "multi_step"
        n_samples:    M in Self-Random sampling. M=1 = single greedy.
        aggregation:  "consistency" | "avg_conf" | "pair_rank"
        topk_k:       K for top-k prompting (only used if prompting="topk")

    Returns dict with at minimum:
        answer, confidence, majority_share, n_samples, n_total, raw,
        total_generation_tokens, execution_time_seconds.
    """
    if prompting == "vanilla":
        prompt = VANILLA_PROMPT.format(answer_type=ANSWER_TYPE, question=question)
        if max_new_tokens is None:
            max_new_tokens = 65
    elif prompting == "cot":
        prompt = COT_PROMPT.format(answer_type=ANSWER_TYPE, question=question)
        if max_new_tokens is None:
            max_new_tokens = 512 # was 265 — too short for GSM8K multi-step
    elif prompting == "topk":
        prompt = TOPK_PROMPT.format(k=topk_k, question=question)
        if max_new_tokens is None:
            max_new_tokens = 265
    elif prompting == "multi_step":
        prompt = MULTI_STEP_PROMPT.format(answer_type=ANSWER_TYPE, question=question)
        if max_new_tokens is None:
            max_new_tokens = 384
    else:
        raise ValueError(f"unknown prompting={prompting}")

    raws, token_counts, elapsed = _sample_raw(
        model,
        tok,
        prompt,
        device,
        n_samples=n_samples,
        temperature=temperature if n_samples > 1 else 0.0,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )

    total_tokens = sum(token_counts)

    if prompting == "topk":
        samples_topk = [_parse_topk(r) for r in raws]
        n_parsed = sum(1 for s in samples_topk if s)
        result = _topk_to_majority(samples_topk)
    else:
        parsed = []
        for r in raws:
            ans, conf = _parse_answer_conf(r)
            if ans is not None and conf is not None:
                parsed.append((ans, conf))
        n_parsed = len(parsed)
        agg_fn = _AGGREGATORS.get(aggregation)
        if agg_fn is None:
            raise ValueError(f"unknown aggregation={aggregation}")
        result = agg_fn(parsed)

    result.update(
        {
            "prompting": prompting,
            "aggregation": aggregation if prompting != "topk" else "topk_avg",
            "n_samples": n_parsed,
            "n_total": n_samples,
            "raw": raws,
            "total_generation_tokens": total_tokens,
            "execution_time_seconds": elapsed,
        }
    )
    return result


def run_xiong_grid(
    model,
    tok,
    question: str,
    device,
    n_samples: int = 5,
    include_topk: bool = True,
    include_multi_step: bool = False,
) -> Dict[str, Dict]:
    """Run the standard cells of the framework. Returns {config_name: result}.

    Each result dict now includes:
        total_generation_tokens : sum of newly generated tokens
        execution_time_seconds  : wall‑clock time for generation

    Default skips multi_step because it's slowest and rarely best.
    """
    out = {}
    out["vanilla_M1"] = run_xiong(
        model,
        tok,
        question,
        device,
        prompting="vanilla",
        n_samples=1,
        aggregation="avg_conf",
    )
    out["cot_M1"] = run_xiong(
        model,
        tok,
        question,
        device,
        prompting="cot",
        n_samples=1,
        aggregation="avg_conf",
    )

    for prompt_kind in ["vanilla", "cot"]:
        for agg in ["consistency", "avg_conf", "pair_rank"]:
            key = f"{prompt_kind}_M{n_samples}_{agg}"
            out[key] = run_xiong(
                model,
                tok,
                question,
                device,
                prompting=prompt_kind,
                n_samples=n_samples,
                aggregation=agg,
            )

    if include_topk:
        out["topk_M1"] = run_xiong(
            model, tok, question, device, prompting="topk", n_samples=1
        )
        out[f"topk_M{n_samples}"] = run_xiong(
            model, tok, question, device, prompting="topk", n_samples=n_samples
        )

    if include_multi_step:
        out["multi_step_M1"] = run_xiong(
            model,
            tok,
            question,
            device,
            prompting="multi_step",
            n_samples=1,
            aggregation="avg_conf",
        )
        out[f"multi_step_M{n_samples}_avg_conf"] = run_xiong(
            model,
            tok,
            question,
            device,
            prompting="multi_step",
            n_samples=n_samples,
            aggregation="avg_conf",
        )

    return out

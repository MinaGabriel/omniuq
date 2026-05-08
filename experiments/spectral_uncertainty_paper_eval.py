# experiments/spectral_uncertainty_paper_eval.py

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from omniuq import DatasetLoader, SpectralUncertainty


# Paper config (Walha et al. 2025, Section 5)
N_SAMPLES = 300
GREEDY_TEMP = 0.1


@torch.inference_mode()
def _greedy_answer(uq: SpectralUncertainty, question: str) -> str:
    # Best-effort answer at t=0.1 used to derive the correctness label.
    # Reuses the tokenizer + model already loaded inside uq.
    inputs = uq.tokenizer(f"Question: {question}\nAnswer:", return_tensors="pt").to(uq.model.device)
    out = uq.model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=True,
        temperature=GREEDY_TEMP,
        pad_token_id=uq.tokenizer.pad_token_id,
        eos_token_id=uq.tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return uq.tokenizer.decode(gen, skip_special_tokens=True).strip().split("\n")[0].strip()


def run(uq: SpectralUncertainty, n_samples: int = N_SAMPLES) -> list[dict]:
    data = DatasetLoader("triviaqa", n_samples=n_samples).load()

    rows = []
    for ex in tqdm(data, desc="TriviaQA"):
        pred = _greedy_answer(uq, ex["question"])
        correct = uq.judge(ex["question"], pred, ex["answers"])
        rows.append({
            "correct": correct,
            **uq.score(ex["question"]),
            "question": ex["question"],
            "prediction": pred,
            "gold": ex["answers"],
        })

    # Report — high uncertainty should flag incorrect answers
    y_true = np.array([1 - r["correct"] for r in rows])
    total = np.array([r["total"] for r in rows])
    epistemic = np.array([r["epistemic"] for r in rows])
    aleatoric = np.array([r["aleatoric"] for r in rows])

    print(f"\n{'='*70}")
    print(f"N: {len(rows)}   Accuracy: {1 - y_true.mean():.4f}")
    print(f"{'='*70}")
    print(f"AUROC (total):     {roc_auc_score(y_true, total)*100:6.2f}%")
    print(f"AUROC (epistemic): {roc_auc_score(y_true, epistemic)*100:6.2f}%")
    print(f"AUROC (aleatoric): {roc_auc_score(y_true, aleatoric)*100:6.2f}%")
    print(f"AUPR  (total):     {average_precision_score(y_true, total)*100:6.2f}%")
    print(f"{'='*70}")
    print(f"Paper Phi-4 14B:   AUROC 91.92%   AUPR 80.79%")
    print(f"Paper LLaMA 4:     AUROC 84.82%   AUPR 60.84%")
    print(f"{'='*70}")

    return rows
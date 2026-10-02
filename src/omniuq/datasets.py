# src/omniuq/datasets.py

from __future__ import annotations

import random

from datasets import load_dataset


class DatasetLoader:
    """Unified loader for QA benchmarks.

    TriviaQA returns a list of {id, question, gold, answers} dicts.
    GSM8K returns a list of {question, answer} dicts.

    Args:
        name: "triviaqa" or "gsm8k".
        split: dataset split (TriviaQA only; the public test split has no answers).
        n_samples: number of questions to keep after shuffling (None = all).
        seed: shuffle seed.
        dedupe: TriviaQA only. rc.nocontext repeats questions (17,944 rows, 9,960 unique ids).
            True keeps one row per question_id and shuffles with Python's random.Random(seed).
            False reproduces the previous behaviour (no de-duplication, datasets' shuffle).
    """

    def __init__(
        self,
        name: str,
        split: str = "validation",
        n_samples: int | None = None,
        seed: int = 42,
        dedupe: bool = True,
    ):
        self.name = name.lower()
        self.split = split
        self.n_samples = n_samples
        self.seed = seed
        self.dedupe = dedupe

    def load(self) -> list[dict]:
        if self.name == "triviaqa":
            return self._load_triviaqa()
        if self.name == "gsm8k":
            return self._load_gsm8k()
        raise ValueError(f"Unknown dataset: {self.name}")

    def _load_triviaqa(self) -> list[dict]:
        # rc.nocontext = closed-book setup matching Walha et al. 2025
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=self.split)

        if not self.dedupe:
            if self.n_samples is not None:
                ds = ds.shuffle(seed=self.seed).select(range(self.n_samples))
            return [self._triviaqa_record(ex) for ex in ds]

        # One row per question, in dataset order, then a seeded shuffle
        seen, pool = set(), []
        for ex in ds:
            if ex["question_id"] in seen:
                continue
            seen.add(ex["question_id"])
            pool.append(self._triviaqa_record(ex))
        random.Random(self.seed).shuffle(pool)
        return pool if self.n_samples is None else pool[: self.n_samples]

    @staticmethod
    def _triviaqa_record(ex: dict) -> dict:
        a = ex["answer"]
        return {
            "id": ex["question_id"],
            "question": ex["question"].strip(),
            "gold": a["value"],
            # All gold spellings, de-duplicated, order preserved
            "answers": list(dict.fromkeys(a["aliases"] + a["normalized_aliases"] + [a["value"]])),
        }

    def _load_gsm8k(self) -> list[dict]:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        if self.n_samples is not None:
            ds = ds.shuffle(seed=self.seed).select(range(self.n_samples))
        return [
            {
                "question": ex["question"],
                # GSM8K answers end with "#### 42" — extract just the number
                "answer": ex["answer"].split("####")[-1].strip(),
            }
            for ex in ds
        ]
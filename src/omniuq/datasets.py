# src/omniuq/datasets.py

from __future__ import annotations

from datasets import load_dataset


class DatasetLoader:
    """Unified loader for QA benchmarks. Returns list of {question, answers} dicts."""

    def __init__(self, name: str, split: str = "validation", n_samples: int | None = None, seed: int = 42):
        self.name = name.lower()
        self.split = split
        self.n_samples = n_samples
        self.seed = seed

    def load(self) -> list[dict]:
        if self.name == "triviaqa":
            return self._load_triviaqa()
        raise ValueError(f"Unknown dataset: {self.name}")

    def _load_triviaqa(self) -> list[dict]:
        # rc.nocontext = closed-book setup matching Walha et al. 2025
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=self.split)
        if self.n_samples is not None:
            ds = ds.shuffle(seed=self.seed).select(range(self.n_samples))
        return [
            {
                "question": ex["question"],
                "answers": ex["answer"]["aliases"] + [ex["answer"]["value"]],
            }
            for ex in ds
        ]
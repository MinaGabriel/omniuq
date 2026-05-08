# omniuq

State-of-the-art uncertainty quantification methods
for large language models.

`omniuq` brings together rigorous, paper-faithful
implementations of methods that measure when an
LLM is unsure and *why*.

---

## Install

```bash
pip install omniuq
```

For low-VRAM setups (e.g. Phi-4 14B on a 24 GB
card), enable quantization:

```bash
pip install "omniuq[quantize]"
```

You'll need an OpenAI API key for the clarifier
and judge:

```bash
export OPENAI_API_KEY=sk-...
```

---

## AU / EU

Every uncertain LLM answer has two possible causes.
`omniuq` separates them.

**Aleatoric Uncertainty (AU)** — uncertainty from
the **input itself**. The question is ambiguous,
underspecified, or has multiple valid
interpretations. Cannot be reduced by a stronger
model.

> *"Who won the World Series?"* — high AU.
> Depends on year, league, team vs. player.

**Epistemic Uncertainty (EU)** — uncertainty from
the **model's lack of knowledge**. The question is
clear; the model just doesn't know. Can be reduced
with retrieval, fine-tuning, or a stronger model.

> *"What is the capital of Wakanda?"* — high EU.
> Question is clear; the model has no real answer.

**Total** = AU + EU.

### Methods

| Method | Decomposes | Paper | Code | Reproduced | Status |
|---|---|---|---|---|---|
| Spectral Uncertainty (Walha et al., AAAI 2026) | AU + EU | [arXiv](https://arxiv.org/abs/2509.22272) | [GitHub](https://github.com/MLO-lab/spectral_uncertainty_decomposition) | TriviaQA: AUROC **89.66%** vs. paper 91.92% — [Colab](https://colab.research.google.com/drive/1VjD4nFdvZR1ad1Z32qU43sGtvCVwdKsD?usp=sharing) | ✅ Available |

### Demo 1 — Spectral Uncertainty

Three ways to run it.

#### Paper-faithful

Phi-4 14B as target, GPT-4o as clarifier,
GPT-4.1 as judge — exactly the paper's setup.

```python
import os
from omniuq import (
    SpectralUncertainty,
    load_llm_model,
    load_openai_client,
)

tokenizer, model = load_llm_model("microsoft/phi-4")

clarifier = load_openai_client(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o",
)
judge = load_openai_client(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4.1",
)

uq = SpectralUncertainty(
    tokenizer, model,
    clarifier=clarifier,
    judge=judge,
)

print(uq.score("What is the capital of France?"))
```

#### Mixed: local target + OpenAI clarifier/judge

Smaller local model for sampling, GPT-4o for
high-quality clarifications.

```python
import os
from omniuq import (
    SpectralUncertainty,
    load_llm_model,
    load_openai_client,
)

tokenizer, model = load_llm_model(
    "Qwen/Qwen2.5-7B-Instruct"
)

clarifier = load_openai_client(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o",
)
judge = load_openai_client(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4.1",
)

uq = SpectralUncertainty(
    tokenizer, model,
    clarifier=clarifier,
    judge=judge,
)

print(uq.score("What is the capital of France?"))
```

#### Fully local — no API calls

Same HuggingFace model used as target, clarifier,
and judge. No OpenAI key needed.

```python
from omniuq import SpectralUncertainty, load_llm_model

tokenizer, model = load_llm_model(
    "meta-llama/Llama-3.1-8B-Instruct"
)

uq = SpectralUncertainty(
    tokenizer, model,
    clarifier=(tokenizer, model),
    judge=(tokenizer, model),
)

print(uq.score("What is the capital of France?"))
```

Note: smaller open models tend to produce noisier
clarifications than GPT-4o, so AU scores will be
less reliable in fully-local mode.

---

## License

MIT
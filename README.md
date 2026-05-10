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

## The big picture: LLM Reliability

When we say "I want a reliable LLM," we usually
mean three different things at once. Researchers
split the field into three branches that build on
each other.

```text
LLM Reliability
│
├── 1. Uncertainty Quantification (UQ)
│   │
│   ├── What is uncertain?
│   │   ├── Input
│   │   ├── Reasoning
│   │   ├── Parameters / knowledge
│   │   └── Prediction / generated output
│   │
│   ├── Why is it uncertain?
│   │   ├── Aleatoric uncertainty
│   │   └── Epistemic uncertainty
│   │
│   └── How do we measure it?
│       ├── Single-generation methods
│       ├── Multi-generation methods
│       ├── External semantic-model methods
│       ├── Fine-tuning / trainable estimators
│       └── Conformal prediction methods
│
├── 2. Confidence Estimation
│   │
│   ├── Confidence in one answer
│   ├── Probability of generated sequence
│   ├── Verbalized confidence
│   ├── LLM-as-a-judge confidence
│   └── Trainable confidence estimators
│
└── 3. Evaluation
    │
    ├── Ranking quality
    │   ├── Can the score separate right from wrong?
    │   └── Metrics: AUROC, AUARC / AURC
    │
    └── Calibration quality
        ├── Are the confidence numbers truthful?
        └── Metric: ECE
```

**Uncertainty Quantification** asks *is the model
unsure, and why?* The "what" branch locates the
source of uncertainty. The "why" branch separates
ambiguity in the question (aleatoric) from gaps in
the model's knowledge (epistemic). The "how" branch
covers the algorithmic families used to estimate
these signals.

**Confidence Estimation** turns uncertainty into a
single trustworthiness number per answer — a score
the application layer can act on (gate a response,
escalate to a human, abstain).

**Evaluation** is how we judge the previous two.
*Ranking quality* asks whether high-uncertainty
answers really are the wrong ones (AUROC, AURC).
*Calibration quality* asks whether the numbers
themselves are honest — when the model says "90%
sure," is it correct 90% of the time? (ECE).

A method can rank well but be miscalibrated, or be
calibrated but rank poorly. Good UQ requires both.

---

## Categories of UQ Methods

`omniuq` focuses on the **UQ** branch above. Within
UQ, methods can be organized by the algorithmic
family they belong to.

```text
UQ Methods
│
├── 1. Input Uncertainty Methods
│   │
│   ├── 1.1 Prompt clarification
│   │   └── Resolves ambiguity by generating clarified versions of the input
│   │
│   ├── 1.2 Prompt perturbation
│   │   └── Measures stability under small input changes
│   │
│   └── 1.3 In-context sample variation
│       └── Measures sensitivity to examples or demonstrations
│
├── 2. Reasoning Uncertainty Methods
│   │
│   ├── 2.1 Chain-of-thought uncertainty
│   │   └── Measures disagreement across reasoning traces
│   │
│   ├── 2.2 Tree-of-thought uncertainty
│   │   └── Measures uncertainty across explored reasoning paths
│   │
│   ├── 2.3 Topology-based reasoning graphs
│   │   └── Measures structure and stability of reasoning graphs
│   │
│   └── 2.4 Uncertainty-guided reasoning repair
│       └── Uses uncertainty to revise weak or unstable reasoning
│
├── 3. Parameter Uncertainty Methods
│   │
│   ├── 3.1 Bayesian LoRA
│   │   └── Approximates posterior uncertainty over adapter weights
│   │
│   ├── 3.2 LoRA ensembles
│   │   └── Uses multiple fine-tuned adapters as an ensemble
│   │
│   ├── 3.3 Supervised uncertainty estimation
│   │   └── Trains a model to predict its own correctness or confidence
│   │
│   └── 3.4 Uncertainty-aware instruction tuning
│       └── Fine-tunes models to express calibrated uncertainty
│
└── 4. Prediction Uncertainty Methods
    │
    ├── 4.1 Single-Generation Methods
    │   │
    │   ├── 4.1.1 Perplexity
    │   │   └── Higher perplexity often indicates lower confidence
    │   │
    │   ├── 4.1.2 Log probability
    │   │   └── Uses token likelihood as a confidence signal
    │   │
    │   ├── 4.1.3 Entropy
    │   │   └── Measures uncertainty in the token distribution
    │   │
    │   ├── 4.1.4 phi_first
    │   │   └── Uses first-token confidence or entropy from a single decode
    │   │
    │   ├── 4.1.5 Response improbability
    │   │   └── Scores how unlikely the generated response is
    │   │
    │   └── 4.1.6 P(True)
    │       └── Uses the model's verbalized confidence that an answer is true
    │
    ├── 4.2 Multi-Generation Methods
    │   │
    │   ├── 4.2.1 Self-consistency
    │   │   └── Measures agreement across multiple sampled answers
    │   │
    │   ├── 4.2.2 Predictive entropy
    │   │   └── Measures entropy over generated answer distributions
    │   │
    │   ├── 4.2.3 Token-level entropy
    │   │   └── Aggregates uncertainty over generated tokens
    │   │
    │   └── 4.2.4 Conformal prediction
    │       └── Produces prediction sets with coverage guarantees
    │
    └── 4.3 Multi-Generation + External Model Methods
        │
        ├── 4.3.1 Semantic entropy
        │   └── Groups semantically equivalent answers before computing entropy
        │
        ├── 4.3.2 NLI clustering
        │   └── Uses an external NLI model to cluster equivalent answers
        │
        ├── 4.3.3 ICE answer grouping
        │   └── Groups answers conditioned on generated clarifications
        │
        ├── 4.3.4 Pairwise similarity graphs
        │   └── Builds graphs from answer-to-answer similarity
        │
        ├── 4.3.5 Graph degree
        │   └── Measures how centrally connected an answer is
        │
        ├── 4.3.6 Eccentricity
        │   └── Measures how far an answer is from other answers in the graph
        │
        └── 4.3.7 Eigenvalue-based metrics
            └── Computes uncertainty from eigenvalues of similarity matrices
```

The first tree is the *map of the research
landscape*. The second tree is the *map of the
algorithms*. They sit at different levels: the
first tells you what question you're answering;
the second tells you which tool to use.

---
## Methods

The **Category** column refers to the numbered
nodes in the UQ Methods tree above.

| Method | Category | Decomposes | Paper | Code | Reproduced | Status |
|---|---|---|---|---|---|---|
| Spectral Uncertainty (Walha et al., AAAI 2026) | 1.1 & 4.3.7 | AU + EU | [arXiv](https://arxiv.org/abs/2509.22272) | [GitHub](https://github.com/MLO-lab/spectral_uncertainty_decomposition) | TriviaQA: AUROC **89.66%** vs. paper 91.92% — [Colab](https://colab.research.google.com/drive/1VjD4nFdvZR1ad1Z32qU43sGtvCVwdKsD?usp=sharing) | ✅ Available |
| Verbalized Confidence (Xiong et al., ICLR 2024) | 4.1.6 & 4.2.1 | — | [arXiv](https://arxiv.org/abs/2306.13063) | [GitHub](https://github.com/MiaoXiong2320/llm-uncertainty) | GSM8K: AUROC Vanilla **56.23%** → CoT+M5+AvgConf **90.92%** (+34.7 pts) — [Colab](https://colab.research.google.com/drive/1mP8fcSDfuv1dqxrYILMXc3YaFrfBF3IZ?usp=sharing) | ✅ Available |

### Demo 1 — Spectral Uncertainty

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

### Demo 2 — Verbalized Confidence (Xiong et al.)

Llama-3.1-8B as target. No API keys needed —
the model verbalizes its own confidence.

```python
from omniuq import load_llm_model
from omniuq.verbalized_xiong import run_xiong

tokenizer, model = load_llm_model(
    "meta-llama/Llama-3.1-8B-Instruct"
)
device = model.device

# Strongest configuration: CoT + Self-Random M=5 + AvgConf
result = run_xiong(
    model, tokenizer,
    "If 7 cars need 14 hours to complete a task, "
    "how long do 5 cars need?",
    device,
    prompting="cot",
    n_samples=5,
    aggregation="avg_conf",
)
print(result["answer"], result["confidence"])
```

---

## License

MIT
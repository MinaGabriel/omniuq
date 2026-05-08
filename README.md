# omniuq

State-of-the-art uncertainty quantification methods for large language models.

## About

`omniuq` is a Python library that brings together state-of-the-art methods for measuring uncertainty in LLM outputs. It decomposes predictive uncertainty into its sources — aleatoric (input ambiguity) and epistemic (model knowledge gaps) — using rigorous, paper-faithful implementations.

Currently included:

- **Spectral Uncertainty** (Walha et al. 2025) — von Neumann entropy over kernel covariance operators for fine-grained aleatoric/epistemic decomposition.

More methods will be added over time, all under one consistent API.

## Install

```bash
pip install omniuq[openai]
```

Or directly from GitHub:

```bash
pip install "omniuq[openai] @ git+https://github.com/MinaGabriel/omniuq.git"
```

## Quick start

```python
import os
from omniuq import SpectralUncertainty, load_llm_model, load_openai_client

tokenizer, model = load_llm_model("meta-llama/Llama-3.1-8B-Instruct")
clarifier = load_openai_client(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o")
judge = load_openai_client(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4.1")

uq = SpectralUncertainty(tokenizer, model, clarifier=clarifier, judge=judge)
result = uq.score("What is the capital of France?")
print(result)
# {'aleatoric': 0.0, 'epistemic': ~0.0, 'total': ~0.0, ...}
```

## License

MIT
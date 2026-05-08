# src/omniuq/utils.py

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_llm_model(model_name: str, device: str | None = None):
    # Load a HuggingFace causal language model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device or "auto",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model


class OpenAIProvider:
    """Wraps an OpenAI client + a chosen model name into one object.

    Lets users pick the model once at construction time, instead of
    threading it through every method that uses the client.
    """

    def __init__(self, api_key: str, model: str):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("Install with: pip install omniuq[openai]") from e
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, prompt: str, max_tokens: int = 256) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content


def load_openai_client(api_key: str, model: str = "gpt-4o") -> OpenAIProvider:
    # Returns an OpenAIProvider bound to a specific model.
    # Used as either a clarifier or a judge in SpectralUncertainty.
    return OpenAIProvider(api_key=api_key, model=model)


def parametric_answer(
    tokenizer, model, question: str, max_new_tokens: int = 128
) -> str:
    # Generate an answer from the model's parametric knowledge.
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

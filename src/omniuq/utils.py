# src/omniuq/utils.py

from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_llm_model(
    model_name: str,
    device: str | None = None,
    quantize: str | None = None,
):
    """Load a HuggingFace causal LM and tokenizer.

    Args:
        model_name: HuggingFace model id (e.g. "microsoft/phi-4").
        device: device map (default: "auto").
        quantize: None for full bf16/fp32, "4bit" for nf4 quantization,
        or "8bit" for int8. Both quantization modes require bitsandbytes.

    Examples:
        # Full precision (best quality, most VRAM)
        tokenizer, model = load_llm_model("microsoft/phi-4")

        # 4-bit (fits Phi-4 14B in ~10GB, slight quality loss)
        tokenizer, model = load_llm_model("microsoft/phi-4", quantize="4bit")

        # 8-bit (middle ground, ~14GB for Phi-4)
        tokenizer, model = load_llm_model("microsoft/phi-4", quantize="8bit")
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    kwargs = {"device_map": device or "auto"}

    if quantize is None:
        kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    elif quantize in {"4bit", "8bit"}:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as e:
            raise ImportError("Install with: pip install omniuq[quantize]") from e

        if quantize == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:  # 8bit
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"quantize must be None, '4bit', or '8bit', got {quantize!r}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model


class OpenAIProvider:
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
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content

def load_openai_client(api_key: str, model: str = "gpt-4o") -> OpenAIProvider:
    return OpenAIProvider(api_key=api_key, model=model)


# Markers that indicate a model has started rambling beyond the answer.
# Comprehensive list covering known artifacts across model families.
ARTIFACT_MARKERS = [
    "\n",
    "#",  # Phi-4: ### problem, ## query, # student
    "Question:",  # textbook-style continuation
    "Q:",
    "Note:",
    "Explanation:",
    "Reasoning:",
    "Step ",  # Phi-4: "Step 1: Identify the..."
    "Alice:",
    "Bob:",  # Phi-4 dialog speakers
    "Student:",
    "Teacher:",
    "Problem:",
    "Query:",
    "<|",
    ">>>",  # leaked special tokens
    "I hope",  # Mistral's "I hope this helps!"
    "Final answer:",  # rambling-then-restating
]


def clean_answer(text: str) -> str:
    """Strip artifacts from a model's raw output. Use after every generation."""
    text = text.strip()
    # Split on rambling artifact markers (newlines, leaked tokens, etc.)
    for marker in ARTIFACT_MARKERS:
        if marker in text:
            text = text.split(marker)[0]
    text = text.strip()
    # Strip TriviaQA/OpenNQ-style generated prefixes (e.g. "A: Paris", "Random guess: Rome")
    for pattern in [
        re.compile(r"The question is unclear\. Random guess:\s*", re.IGNORECASE),
        re.compile(r"Random guess:\s*", re.IGNORECASE),
        re.compile(r"^A:\s*", re.IGNORECASE),
    ]:
        m = pattern.match(text)
        if m:
            text = text[m.end():].strip()
            break
    # Strip surrounding quotes and trailing punctuation
    text = text.strip("\"'")
    text = re.sub(r"[.?!]+$", "", text)
    return text.strip()


def build_prompt(tokenizer, question: str, terse: bool = True) -> str:
    """Build a model-appropriate prompt.

    Uses the chat template if the tokenizer has one (modern instruct models).
    Falls back to completion-style prompting for older models.
    """
    user_content = question
    if terse:
        user_content = (
            f"{question}\n\nAnswer with a single short phrase. No explanation."
        )

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": user_content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback for base models without a chat template
    return f"Question: {user_content}\nAnswer:"


def generate_answers(
    tokenizer,
    model,
    question: str,
    n: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 100,
    terse: bool = True,
    top_p: float | None = None,
    seed: int | None = None,
) -> list[str]:
    """One generation function used everywhere — same prompt logic across the library."""
    prompt = build_prompt(tokenizer, question, terse=terse)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    if seed is not None:
        torch.manual_seed(seed)

    do_sample = temperature > 0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["num_return_sequences"] = n
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    if do_sample:
        gen = outputs[:, inputs["input_ids"].shape[1] :]
        return [
            clean_answer(tokenizer.decode(g, skip_special_tokens=True)) for g in gen
        ]
    else:
        gen = outputs[0][inputs["input_ids"].shape[1] :]
        return [clean_answer(tokenizer.decode(gen, skip_special_tokens=True))]


def parametric_answer(
    tokenizer, model, question: str, max_new_tokens: int = 128
) -> str:
    """Single greedy answer — kept for backwards compatibility."""
    return generate_answers(
        tokenizer,
        model,
        question,
        n=1,
        temperature=0.0,
        max_new_tokens=max_new_tokens,
        terse=False,
    )[0]

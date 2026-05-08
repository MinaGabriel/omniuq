# src/omniuq/runtime.py

from __future__ import annotations

import os
import logging
import warnings


def configure_runtime():
    # Configure HuggingFace runtime and suppress common warning noise.

    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/hf_cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)
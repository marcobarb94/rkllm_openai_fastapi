from typing import Collection, List, Literal
import tiktoken
from jinja2 import Template
import json
import numpy as np


def num_tokens_from_string(string: str,
                           encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string, allowed_special={"<|endoftext|>"}))


def parse_message_to_prompt(messages: List[str],
                            tokenizer_config: dict) -> str:

    ct = tokenizer_config['chat_template']
    template = Template(ct)
    return template.render(messages=messages)


def get_sentence_embedding(float_array: Collection[float],
                           embed_size: int,
                           emb_type: Literal["mean", "max", "last"] = "mean"):
    """Quale metodo usare? ✔ Se vuoi semplicità → Mean pooling ✔ Se vuoi cogliere le parti dominanti → Max pooling ✔ Se il testo è breve e importante → Ultimo token ✔ Se vuoi massima qualità → Concatenazione o PCA
    """
    hidden_states = np.array(float_array).reshape(
        len(float_array) // embed_size, embed_size)
    match emb_type:
        case "mean":
            _emb = np.mean(hidden_states, axis=0)
        case "max":
            _emb = np.amax(hidden_states, axis=0)
        case "last":
            _emb = hidden_states[-1, :]
    return _emb / np.linalg.norm(_emb)

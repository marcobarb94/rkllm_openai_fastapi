import logging
from typing import Any, Collection, List, Literal, Optional, Self
import tiktoken
from jinja2 import Template
import json
import numpy as np


def num_tokens_from_string(string: str,
                           encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string, allowed_special={"<|endoftext|>"}))


def raise_exception(nome):
    logging.error(f"Template error {nome}")
    return f"\nScrivi all'utente che c'è questo errore: {nome}!\n"


def parse_message_to_prompt(messages: List[str],
                            tokenizer_config: dict,
                            enable_thinking: bool = False) -> str:

    ct = tokenizer_config['chat_template']
    template = Template(ct)
    tokenizer_config['enable_thinking'] = enable_thinking
    return template.render(**tokenizer_config,
                           messages=messages,
                           raise_exception=raise_exception)


class EmbeddingsLLMData:
    token_size: int
    hidden_layer: np.ndarray[float, float]
    logits: np.ndarray[float]

    def __init__(self,
                 logits: Optional[Collection[float]] = None,
                 emb_vocab_size: Optional[int] = None,
                 hidden_layer: Optional[Collection[float]] = None):
        if logits is not None:
            self.token_size = len(logits) // emb_vocab_size

            self.logits = np.array(logits).reshape(self.token_size,
                                                   emb_vocab_size)
        if hidden_layer is not None:
            self.token_size = len(hidden_layer) // emb_vocab_size
            self.hidden_layer = np.array(hidden_layer).reshape(
                self.token_size, emb_vocab_size)

    def merge(self, other: Self):
        if other.logits is not None:
            self.logits = other.logits
        if other.hidden_layer is not None:
            self.hidden_layer = other.hidden_layer
            self.token_size = other.token_size


class QueueResult:
    queue_id: int
    payload: Optional[EmbeddingsLLMData | str | bool | Any] = None

    def __init__(self, queue_id: int, payload: EmbeddingsLLMData | str):
        self.queue_id = queue_id
        self.payload = payload

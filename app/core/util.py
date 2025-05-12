from typing import List
import tiktoken
from jinja2 import Template
import json


def num_tokens_from_string(string: str,
                           encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string, allowed_special={"<|endoftext|>"}))


def parse_message_to_prompt(messages: List[str], tokenizer_config: dict) -> str:

    ct = tokenizer_config['chat_template']
    template = Template(ct)
    return template.render(messages=messages)

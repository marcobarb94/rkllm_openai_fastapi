import ctypes

from pydantic import BaseModel


class Token(ctypes.Structure):
    _fields_ = [
        ("logprob", ctypes.c_float),
        ("id", ctypes.c_int32)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("tokens", ctypes.POINTER(Token)),
        ("num", ctypes.c_int32)
    ]


class LLMParams(BaseModel):
    max_context_len: int = 320
    max_new_tokens: int = 512
    top_k: int = 1
    top_p: float = 0.9
    temperature: float = 0.8
    repeat_penalty: float = 1.1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    mirostat: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    logprobs: bool = False
    top_logprobs: int = 5
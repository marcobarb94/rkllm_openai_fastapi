import ctypes

from app.core.entities_llm import LLMParams, RKLLMResult

from typing import Optional

from app.core.rkllmserver import callback

# Define
# Define types for better readability and type checking
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

# Load the shared library
rkllm_lib = ctypes.CDLL('lib/librkllmrt.so')

DEFAULT_PROMPT_TEXT_PREFIX = "<|im_start|>system You are a helpful assistant. <|im_end|> <|im_start|>user"
DEFAULT_PROMPT_TEXT_POSTFIX = "<|im_end|><|im_start|>assistant"


class RKNNllmParam(ctypes.Structure):
    _fields_ = [("model_path", ctypes.c_char_p),
                ("num_npu_core", ctypes.c_int32),
                ("max_context_len", ctypes.c_int32),
                ("max_new_tokens", ctypes.c_int32), ("top_k", ctypes.c_int32),
                ("top_p", ctypes.c_float), ("temperature", ctypes.c_float),
                ("repeat_penalty", ctypes.c_float),
                ("frequency_penalty", ctypes.c_float),
                ("presence_penalty", ctypes.c_float),
                ("mirostat", ctypes.c_int32), ("mirostat_tau", ctypes.c_float),
                ("mirostat_eta", ctypes.c_float), ("logprobs", ctypes.c_bool),
                ("top_logprobs", ctypes.c_int32), ("use_gpu", ctypes.c_bool)]


callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult),
                                 ctypes.c_void_p, ctypes.c_int)
c_callback = callback_type(callback)


class RKLLM(object):
    def __init__(self,
                 model_path: str,
                 target_platform: str = "rk3588",
                 prompt_text_prefix: str = DEFAULT_PROMPT_TEXT_PREFIX,
                 prompt_text_postfix: str = DEFAULT_PROMPT_TEXT_POSTFIX,
                 llm_params: Optional[LLMParams] = None):
        if llm_params is None:
            llm_params = LLMParams()
        rknnllm_param = RKNNllmParam()
        rknnllm_param.model_path = bytes(model_path, 'utf-8')

        if target_platform == "rk3588":
            rknnllm_param.num_npu_core = 3
        elif target_platform == "rk3576":
            rknnllm_param.num_npu_core = 1

        # Initialize other parameters
        rknnllm_param.max_context_len = llm_params.max_context_len
        rknnllm_param.max_new_tokens = llm_params.max_new_tokens
        rknnllm_param.top_k = llm_params.top_k
        rknnllm_param.top_p = llm_params.top_p
        rknnllm_param.temperature = llm_params.temperature
        rknnllm_param.repeat_penalty = llm_params.repeat_penalty
        rknnllm_param.frequency_penalty = llm_params.frequency_penalty
        rknnllm_param.presence_penalty = llm_params.presence_penalty
        rknnllm_param.mirostat = llm_params.mirostat
        rknnllm_param.mirostat_tau = llm_params.mirostat_tau
        rknnllm_param.mirostat_eta = llm_params.mirostat_eta
        rknnllm_param.logprobs = llm_params.logprobs
        rknnllm_param.top_logprobs = llm_params.top_logprobs
        rknnllm_param.use_gpu = True

        self.handle = RKLLM_Handle_t()

        # Set up the function prototypes with type hints
        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t),
            ctypes.POINTER(RKNNllmParam), callback_type
        ]
        self.rkllm_init.restype = ctypes.c_int

        # Initialize the RKLLM handle
        self.rkllm_init(ctypes.byref(self.handle), rknnllm_param, c_callback)

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [
            RKLLM_Handle_t,
            ctypes.POINTER(ctypes.c_char), ctypes.c_void_p
        ]
        self.rkllm_run.restype = ctypes.c_int

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        self.prompt_text_prefix = prompt_text_prefix
        self.prompt_text_postfix = prompt_text_postfix

    def run(self, prompt: str) -> None:
        # Construct the full prompt with prefix and postfix
        full_prompt = bytes(
            self.prompt_text_prefix + prompt + self.prompt_text_postfix,
            'utf-8')

        # Run the model
        self.rkllm_run(self.handle, full_prompt, ctypes.byref(userdata))

    def release(self) -> None:
        # Destroy the RKLLM handle
        self.rkllm_destroy(self.handle)

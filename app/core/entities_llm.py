import ctypes
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL = 0
LLMCallState.RKLLM_RUN_WAITING = 1
LLMCallState.RKLLM_RUN_FINISH = 2
LLMCallState.RKLLM_RUN_ERROR = 3

RKLLMInputMode = ctypes.c_int
RKLLMInputMode.RKLLM_INPUT_PROMPT = 0
RKLLMInputMode.RKLLM_INPUT_TOKEN = 1
RKLLMInputMode.RKLLM_INPUT_EMBED = 2
RKLLMInputMode.RKLLM_INPUT_MULTIMODAL = 3

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0
RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLMInferMode.RKLLM_INFER_GET_LOGITS = 2

RKLLMInputType = ctypes.c_int
RKLLMInputType.RKLLM_INPUT_PROMPT = 0,  # < Input is a text prompt. */
RKLLMInputType.RKLLM_INPUT_TOKEN = 1,  # < Input is a sequence of tokens. */
RKLLMInputType.RKLLM_INPUT_EMBED = 2,  # < Input is an embedding vector. */
RKLLMInputType.RKLLM_INPUT_MULTIMODAL = 3,  # < Input is multimodal (e.g., text and image). */


class RKLLMExtendParam(ctypes.Structure):
    """
        typedef struct {
            int32_t      base_domain_id;         # < base_domain_id */
            int8_t       embed_flash;            # < Indicates whether to query word embedding vectors from flash memory (1) or not (0). */
            int8_t       enabled_cpus_num;       # < Number of CPUs enabled for inference. */
            uint32_t     enabled_cpus_mask;      # < Bitmask indicating which CPUs to enable for inference. */
            uint8_t      n_batch;                # < Number of input samples processed concurrently in one forward pass. Set to >1 to enable batched inference. Default is 1. */
            int8_t       use_cross_attn;         # < Whether to enable cross attention (non-zero to enable, 0 to disable). */
            uint8_t      reserved[104];          # < reserved */
        } RKLLMExtendParam;
        """
    _fields_ = [("base_domain_id", ctypes.c_int32),
                ("embed_flash", ctypes.c_int8),
                ("enabled_cpus_num", ctypes.c_int8),
                ("enabled_cpus_mask", ctypes.c_uint32),
                ("n_batch", ctypes.c_uint8), ("use_cross_attn", ctypes.c_int8),
                ("reserved", ctypes.c_uint8 * 104)]


class RKLLMParam(ctypes.Structure):
    """"typedef struct {
    const char* model_path;          # < Path to the model file. */
    int32_t max_context_len;         # < Maximum number of tokens in the context window. */
    int32_t max_new_tokens;          # < Maximum number of new tokens to generate. */
    int32_t top_k;                   # < Top-K sampling parameter for token generation. */
    int32_t n_keep;                  #  number of kv cache to keep at the beginning when shifting context window */
    float top_p;                     # < Top-P (nucleus) sampling parameter. */
    float temperature;               # < Sampling temperature, affecting the randomness of token selection. */
    float repeat_penalty;            # < Penalty for repeating tokens in generation. */
    float frequency_penalty;         # < Penalizes frequent tokens during generation. */
    float presence_penalty;          # < Penalizes tokens based on their presence in the input. */
    int32_t mirostat;                # < Mirostat sampling strategy flag (0 to disable). */
    float mirostat_tau;              # < Tau parameter for Mirostat sampling. */
    float mirostat_eta;              # < Eta parameter for Mirostat sampling. */
    bool skip_special_token;         # < Whether to skip special tokens during generation. */
    bool is_async;                   # < Whether to run inference asynchronously. */
    const char* img_start;           # < Starting position of an image in multimodal input. */
    const char* img_end;             # < Ending position of an image in multimodal input. */
    const char* img_content;         # < Pointer to the image content. */
    RKLLMExtendParam extend_param;  # < Extend parameters. */
} RKLLMParam;"""
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [("lora_adapter_path", ctypes.c_char_p),
                ("lora_adapter_name", ctypes.c_char_p),
                ("scale", ctypes.c_float)]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [("embed", ctypes.POINTER(ctypes.c_float)),
                ("n_tokens", ctypes.c_size_t)]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [("input_ids", ctypes.POINTER(ctypes.c_int32)),
                ("n_tokens", ctypes.c_size_t)]


class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [("prompt", ctypes.c_char_p),
                ("image_embed", ctypes.POINTER(ctypes.c_float)),
                ("n_image_tokens", ctypes.c_size_t),
                ("n_image", ctypes.c_size_t), ("image_width", ctypes.c_size_t),
                ("image_height", ctypes.c_size_t)]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p),
                ("embed_input", RKLLMEmbedInput),
                ("token_input", RKLLMTokenInput),
                ("multimodal_input", RKLLMMultiModelInput)]


class RKLLMInput(ctypes.Structure):
    # _fields_ = [("input_mode", ctypes.c_int), ("input_data", RKLLMInputUnion)]
    _fields_ = [
        ('role', ctypes.c_char_p),  # const char*
        ('enable_thinking', ctypes.c_bool),  # bool (C: _Bool, 1 byte)
        ('input_type', RKLLMInputType),  # enum come int, 
        ("input_data", RKLLMInputUnion)
    ]


class RKLLMCrossAttnParam(ctypes.Structure):
    _fields_ = [
        ('encoder_k_cache', ctypes.POINTER(ctypes.c_float)),  # float*
        ('encoder_v_cache', ctypes.POINTER(ctypes.c_float)),  # float*
        ('encoder_mask', ctypes.POINTER(ctypes.c_float)),  # float*
        ('encoder_pos', ctypes.POINTER(ctypes.c_int32)),  # int32_t*
        ('num_tokens', ctypes.c_int),  # int (usa c_int32 se l'ABI lo richiede)
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [("save_prompt_cache", ctypes.c_int),
                ("prompt_cache_path", ctypes.c_char_p)]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [("mode", RKLLMInferMode),
                ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
                ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
                ("keep_history", ctypes.c_int)]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [("hidden_states", ctypes.POINTER(ctypes.c_float)),
                ("embd_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [("logits", ctypes.POINTER(ctypes.c_float)),
                ("vocab_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]


class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ('prefill_time_ms', ctypes.c_float),  # float
        ('prefill_tokens', ctypes.c_int),  # int
        ('generate_time_ms', ctypes.c_float),  # float
        ('generate_tokens', ctypes.c_int),  # int
        ('memory_usage_mb', ctypes.c_float),  # float
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [("text", ctypes.c_char_p), ("token_id", ctypes.c_int),
                ("last_hidden_layer", RKLLMResultLastHiddenLayer),
                ("logits", RKLLMResultLogits), ('perf', RKLLMPerfStat)]


class UserdataCallback(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int)]


# TODO: rkllm_clear_kv_cache e rkllm_get_kv_cache_size e rkllm_set_function_tools e rkllm_set_cross_attn_params

class ToolSupport(BaseModel):
    #rkllm_set_function_tools
    system_prompt: str
    tools: str
    tool_response_str : str
    
class LLMParams(BaseModel):
    max_context_len: int = 4096
    max_new_tokens: int = 2048
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
    tools: Optional[ToolSupport] = None


class EngineComunication(BaseModel):
    function_name: Literal["run", "abort_job", "is_running"]
    params: Dict[str, Any]



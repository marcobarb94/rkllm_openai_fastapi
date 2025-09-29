import contextlib
import ctypes
import multiprocessing
import sys
import queue
import threading
import logging

from typing import Any, Collection, Literal, Optional, Tuple

from core.util import EmbeddingsLLMData, QueueResult
from core.entities_llm import *

# nm -D rkllm_server/lib/librkllmrt.so
"""
00000000000b8f80 T rkllm_abort
00000000000b9640 T rkllm_accuracy_analysis
00000000000b8fa0 T rkllm_clear_kv_cache
00000000000b8fc0 T rkllm_createDefaultParam
00000000000b8f20 T rkllm_destroy
00000000000b8e40 T rkllm_init
00000000000b8f90 T rkllm_is_running
00000000000b8ef0 T rkllm_load_lora
00000000000b8f00 T rkllm_load_prompt_cache
00000000001394e0 T rkllm_print_memorys
00000000001392d0 T rkllm_print_timings
00000000000b8f10 T rkllm_release_prompt_cache
00000000000b8f60 T rkllm_run
00000000000b8f70 T rkllm_run_async
00000000000b8fb0 T rkllm_set_chat_template
"""
# nm -D rkllm_server/lib/librkllmrt.so | c++filt
# https://github.com/airockchip/rknn-llm/blob/main/rkllm-runtime/Linux/librkllm_api/include/rkllm.h

# Define the structures from the library
RKLLM_Handle_t = ctypes.c_void_p
# userdata = ctypes.c_void_p(None)

# Create a lock to control multi-user access to the server.
lock = threading.Lock()

# Create a global variable to indicate whether the server is currently in a blocked state.
is_blocking = False

# Define global variables to store the callback function output for displaying in the Gradio interface
result_queue = multiprocessing.Queue()
control_queue = multiprocessing.Queue()
cmd_queue: 'multiprocessing.Queue[EngineComunication]' = multiprocessing.Queue(
)
callback_queue = multiprocessing.Queue()

global_state = -1
split_byte_data = bytes(b"")  # Used to store the segmented byte data


def pointer_to_pyth_float(
        pointer: Any,
        len_data: int,
        data_type: ctypes = ctypes.c_float) -> Collection[Any]:
    data_size = len_data * ctypes.sizeof(data_type)
    logging.info(f"data_size: {data_size}")
    data = ctypes.cast(pointer, ctypes.POINTER(data_type))
    float_array_type = data_type * (data_size // ctypes.sizeof(data_type))
    return float_array_type.from_address(ctypes.addressof(data.contents))


# Define the callback function
def callback_impl(result: RKLLMResult, userdata_ptr: ctypes.POINTER, state: LLMCallState):
    userdata = ctypes.cast(userdata_ptr, ctypes.POINTER(UserdataCallback)).contents
    global result_queue, global_state, callback_queue
    if state == LLMCallState.RKLLM_RUN_FINISH:
        global_state = state
        result_queue.put(
            QueueResult(queue_id=userdata.id, payload=result.contents.perf))
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        global_state = state
        result_queue.put(QueueResult(queue_id=userdata.id, payload=False))
        logging.error("run error")
    elif state == LLMCallState.RKLLM_RUN_NORMAL:
        global_state = state
        last_hidden_layer_res = result.contents.last_hidden_layer
        logits_res = result.contents.logits
        if last_hidden_layer_res.embd_size != 0 and last_hidden_layer_res.num_tokens != 0:
            '''
            If using the GET_LAST_HIDDEN_LAYER function, the callback interface will return the memory pointer: last_hidden_layer, the number of tokens: num_tokens, and the size of the hidden layer: embd_size.
            With these three parameters, you can retrieve the data from last_hidden_layer.
            Note: The data needs to be retrieved during the current callback; if not obtained in time, the pointer will be released by the next callback.
            '''
            float_array = pointer_to_pyth_float(
                pointer=last_hidden_layer_res.hidden_states,
                len_data=last_hidden_layer_res.embd_size *
                last_hidden_layer_res.num_tokens)
            result_queue.put(
                QueueResult(queue_id=userdata.id,
                            payload=EmbeddingsLLMData(
                                emb_vocab_size=last_hidden_layer_res.embd_size,
                                hidden_layer=float_array)))
        elif logits_res.vocab_size != 0 and logits_res.num_tokens != 0:
            float_array = pointer_to_pyth_float(
                pointer=logits_res.logits,
                len_data=logits_res.vocab_size * logits_res.num_tokens)
            result_queue.put(
                QueueResult(queue_id=userdata.id,
                            payload=EmbeddingsLLMData(
                                logits=float_array,
                                emb_vocab_size=logits_res.vocab_size)))
            print(result.contents.text.decode('utf-8'))
        else:
            _data = result.contents.text
            result_queue.put(
                QueueResult(
                    queue_id=userdata.id,
                    payload=("" if _data is None else _data.decode('utf-8'))))

    _get = None
    with contextlib.suppress(queue.Empty):
        _get = callback_queue.get(block=False)

    if _get is not None:
        return _get


# Connect the callback function between the Python side and the C++ side
# @return int Return value indicating the handling status:
#  *         - 0: Continue inference normally.
#  *         - 1: Pause inference. If the user wants to modify or intervene in the result (e.g., editing output, injecting new prompt),
#  *              return 1 to suspend the current inference. Later, call `rkllm_run` with updated content to resume inference.
callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult),
                                 ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)

DEFAULT_PROMPT_TEXT_PREFIX = "<|im_start|>user"
DEFAULT_PROMPT_TEXT_POSTFIX = "<|im_end|><|im_start|>assistant"
DEFAULT_SYSTEM_PROMPT = "<|im_start|>system You are a helpful assistant. <|im_end|>"


# Define the RKLLM class, which includes initialization, inference, and release operations for the RKLLM model in the dynamic library
class RKLLM(object):

    def __init__(self,
                 model_path: str,
                 lora_model_path: Optional[str] = None,
                 prompt_cache_path: Optional[str] = None,
                 set_chat_template: bool = False,
                 prompt_text_prefix: str = DEFAULT_PROMPT_TEXT_PREFIX,
                 prompt_text_postfix: str = DEFAULT_PROMPT_TEXT_POSTFIX,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 llm_params: Optional[LLMParams] = None,
                 dll_path: str = '../libs/librkllmrt.so'):

        # Set the dynamic library path
        rkllm_lib = ctypes.CDLL(dll_path)

        self.f_rkllm_createDefaultParam = rkllm_lib.rkllm_createDefaultParam
        self.f_rkllm_createDefaultParam.argtypes = []
        self.f_rkllm_createDefaultParam.restype = RKLLMParam

        rkllm_param = self.f_rkllm_createDefaultParam()

        if llm_params is None:
            llm_params = LLMParams()
        elif isinstance(llm_params, dict):
            llm_params = LLMParams.model_validate(llm_params)
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        # Initialize other parameters
        rkllm_param.max_context_len = llm_params.max_context_len
        rkllm_param.max_new_tokens = llm_params.max_new_tokens
        rkllm_param.top_k = llm_params.top_k
        rkllm_param.top_p = llm_params.top_p
        rkllm_param.temperature = llm_params.temperature
        rkllm_param.repeat_penalty = llm_params.repeat_penalty
        rkllm_param.frequency_penalty = llm_params.frequency_penalty
        rkllm_param.presence_penalty = llm_params.presence_penalty
        rkllm_param.mirostat = llm_params.mirostat
        rkllm_param.mirostat_tau = llm_params.mirostat_tau
        rkllm_param.mirostat_eta = llm_params.mirostat_eta
        rkllm_param.logprobs = llm_params.logprobs
        rkllm_param.top_logprobs = llm_params.top_logprobs
        # rkllm_param.use_gpu = True
        rkllm_param.is_async = False
        rkllm_param.img_start = "".encode('utf-8')
        rkllm_param.img_end = "".encode('utf-8')
        rkllm_param.img_content = "".encode('utf-8')
        rkllm_param.extend_param.base_domain_id = 0

        rkllm_param.extend_param.enabled_cpus_num = 4
        rkllm_param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (
            1 << 6) | (1 << 7)

        rkllm_param.extend_param.n_batch = 1
        rkllm_param.extend_param.use_cross_attn = 0  # metto zero o esplode la ram

        self.handle = RKLLM_Handle_t()

        self.f_rkllm_init = rkllm_lib.rkllm_init
        self.f_rkllm_init.argtypes = [
            ctypes.POINTER(RKLLM_Handle_t),
            ctypes.POINTER(RKLLMParam), callback_type
        ]
        self.f_rkllm_init.restype = ctypes.c_int
        self.f_rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param),
                          callback)

        self.f_rkllm_run = rkllm_lib.rkllm_run
        self.f_rkllm_run.argtypes = [
            RKLLM_Handle_t,
            ctypes.POINTER(RKLLMInput),
            ctypes.POINTER(RKLLMInferParam),
            ctypes.POINTER(UserdataCallback)
        ]
        self.f_rkllm_run.restype = ctypes.c_int

        self.f_set_chat_template = rkllm_lib.rkllm_set_chat_template
        self.f_set_chat_template.argtypes = [
            RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p
        ]
        self.f_set_chat_template.restype = ctypes.c_int

        if set_chat_template:
            _res = self.f_set_chat_template(
                self.handle, ctypes.c_char_p(system_prompt.encode('utf-8')),
                ctypes.c_char_p(prompt_text_prefix.encode('utf-8')),
                ctypes.c_char_p(prompt_text_postfix.encode('utf-8')))
            logging.info(f"Set Chat Template: {_res}")

        self.f_rkllm_destroy = rkllm_lib.rkllm_destroy
        self.f_rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.f_rkllm_destroy.restype = ctypes.c_int

        self.f_rkllm_abort = rkllm_lib.rkllm_abort
        self.f_rkllm_abort.argtypes = [RKLLM_Handle_t]
        self.f_rkllm_abort.restype = ctypes.c_int

        self.f_rkllm_is_running = rkllm_lib.rkllm_is_running
        self.f_rkllm_is_running.argtypes = [RKLLM_Handle_t]
        self.f_rkllm_is_running.restype = ctypes.c_int

        self.f_rkllm_set_function_tools = rkllm_lib.rkllm_set_function_tools
        self.f_rkllm_set_function_tools.argtypes = [
            RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p
        ]
        self.f_rkllm_set_function_tools.restype = ctypes.c_int

        if llm_params.tools is not None:
            _res = self.f_rkllm_set_function_tools(
                self.handle,
                ctypes.c_char_p(
                    llm_params.tools.system_prompt.encode('utf-8')),
                ctypes.c_char_p(llm_params.tools.tools.encode('utf-8')),
                ctypes.c_char_p(
                    llm_params.tools.tool_response_str.encode('utf-8')))
            logging.info(f"Set Fucntion Tools: {_res}")

        rkllm_lora_params = None
        if lora_model_path:
            lora_adapter_name = "test"
            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0,
                          ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = ctypes.c_char_p(
                (lora_model_path).encode('utf-8'))
            lora_adapter.lora_adapter_name = ctypes.c_char_p(
                (lora_adapter_name).encode('utf-8'))
            lora_adapter.scale = 1.0

            self.f_rkllm_load_lora = rkllm_lib.rkllm_load_lora
            self.f_rkllm_load_lora.argtypes = [
                RKLLM_Handle_t,
                ctypes.POINTER(RKLLMLoraAdapter)
            ]
            self.f_rkllm_load_lora.restype = ctypes.c_int
            self.f_rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p(
                (lora_adapter_name).encode('utf-8'))

        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0,
                      ctypes.sizeof(RKLLMInferParam))
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.rkllm_infer_params.lora_params = ctypes.pointer(
            rkllm_lora_params) if rkllm_lora_params else None
        self.rkllm_infer_params.keep_history = 0

        self.prompt_cache_path = None
        if prompt_cache_path:
            self.prompt_cache_path = prompt_cache_path

            self.f_rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            self.f_rkllm_load_prompt_cache.argtypes = [
                RKLLM_Handle_t, ctypes.c_char_p
            ]
            self.f_rkllm_load_prompt_cache.restype = ctypes.c_int
            self.f_rkllm_load_prompt_cache(
                self.handle,
                ctypes.c_char_p((prompt_cache_path).encode('utf-8')))

    def run(self,
            prompt: str,
            infer_type: Literal["generate", "hidden_layer",
                                "logit"] = "generate",
            userdata: Optional[UserdataCallback] = None,
            role: str = 'user',
            enable_thinking: bool = False):
        rkllm_input = RKLLMInput()
        rkllm_input.input_mode = RKLLMInputMode.RKLLM_INPUT_PROMPT
        rkllm_input.role = role.encode(
            'utf-8') if role is not None else "user".encode('utf-8')
        rkllm_input.enable_thinking = ctypes.c_bool(enable_thinking)
        rkllm_input.input_data.prompt_input = ctypes.c_char_p(
            prompt.encode('utf-8'))

        match infer_type:
            case "generate":
                self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
            case "hidden_layer":
                self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER
            case "logit":
                self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GET_LOGITS
        # if userdata:
        #     ctypes.memset(ctypes.byref(userdata), 0,
        #                   ctypes.sizeof(UserdataCallback))

        self.f_rkllm_run(self.handle, ctypes.byref(rkllm_input),
                         ctypes.byref(self.rkllm_infer_params),
                         ctypes.byref(userdata) if userdata else None)
        return

    def release(self):
        self.f_rkllm_destroy(self.handle)

    def abort_job(self) -> bool:
        return self.f_rkllm_abort(self.handle) == 0

    def is_running(self) -> bool:
        # status code (0 if a task is running, non-zero for otherwise).
        return self.f_rkllm_is_running(self.handle) != 0

    def __del__(self) -> None:
        self.abort_job()
        self.release()

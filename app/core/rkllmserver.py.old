import ctypes
import sys
import threading
import time
import json
import tiktoken
import logging
import os
from core.entities_llm import LLMCallState


is_blocking = False

global_text = []
global_state = -1
split_byte_data = bytes(b"")  # Для сохранения разделенных байтовых данных


# Определяем функцию обратного вызова
def callback(result, userdata, state):
    global global_text, global_state, split_byte_data
    if state == LLMCallState.RKLLM_RUN_FINISH:
        global_state = state
        # print("\n")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        global_state = state
        logging.error("run error")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_GET_LAST_HIDDEN_LAYER:
        '''
        If using the GET_LAST_HIDDEN_LAYER function, the callback interface will return the memory pointer: last_hidden_layer, the number of tokens: num_tokens, and the size of the hidden layer: embd_size.
        With these three parameters, you can retrieve the data from last_hidden_layer.
        Note: The data needs to be retrieved during the current callback; if not obtained in time, the pointer will be released by the next callback.
        '''
        if result.last_hidden_layer.embd_size != 0 and result.last_hidden_layer.num_tokens != 0:
            data_size = result.last_hidden_layer.embd_size * result.last_hidden_layer.num_tokens * ctypes.sizeof(ctypes.c_float)
            logging.info(f"data_size: {data_size}")
            global_text.append(f"data_size: {data_size}\n")
            output_path = os.getcwd() + "/last_hidden_layer.bin"
            with open(output_path, "wb") as outFile:
                data = ctypes.cast(result.last_hidden_layer.hidden_states, ctypes.POINTER(ctypes.c_float))
                float_array_type = ctypes.c_float * (data_size // ctypes.sizeof(ctypes.c_float))
                float_array = float_array_type.from_address(ctypes.addressof(data.contents))
                outFile.write(bytearray(float_array))
                logging.info(f"Data saved to {output_path} successfully!")
                global_text.append(f"Data saved to {output_path} successfully!")
        else:
            logging.error("Invalid hidden layer data.")
            global_text.append("Invalid hidden layer data.")
        global_state = state
        time.sleep(0.05) # Delay for 0.05 seconds to wait for the output result
        sys.stdout.flush()
    else:
        # Save the output token text and the RKLLM running state
        global_state = state
        # Monitor if the current byte data is complete; if incomplete, record it for later parsing
        try:
            global_text.append((split_byte_data + result.contents.text).decode('utf-8'))
            logging.info((split_byte_data + result.contents.text).decode('utf-8'), end='')
            split_byte_data = bytes(b"")
        except:
            if result.contents.text != None:
                split_byte_data += result.contents.text
        sys.stdout.flush()

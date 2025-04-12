import ctypes
import sys
import threading
import time
import json
import tiktoken
import logging


is_blocking = False

global_text = []
global_state = -1
split_byte_data = bytes(b"")  # Для сохранения разделенных байтовых данных


# Определяем функцию обратного вызова
def callback(result, userdata, state):
    global global_text, global_state, split_byte_data
    if state == 0:
        # Сохраняем выходной текст токена и состояние выполнения RKLLM
        global_state = state
        # Проверяем целостность текущих байтовых данных, если неполные - записываем для последующего анализа
        try:
            global_text.append((split_byte_data + result.contents.text).decode('utf-8'))
            logging.info((split_byte_data + result.contents.text).decode('utf-8'), end='')
            split_byte_data = bytes(b"")
        except:
            split_byte_data += result.contents.text
        sys.stdout.flush()
    elif state == 1:
        # Сохраняем состояние выполнения RKLLM
        global_state = state
        logging.info("\n")
        sys.stdout.flush()
    else:
        logging.error("execution error")







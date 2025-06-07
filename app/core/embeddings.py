import logging
import threading
from time import sleep
from typing import Generator
import numpy as np
from core.rkllm import global_text, global_state


def generate_embeddings(text: str, rkllm_model) -> np.array:
    global global_text, global_state

    global_text._init(4096)
    global_state = -1

    prompt = text.strip()

    rkllm_output = []

    model_thread = threading.Thread(target=rkllm_model.run,
                                    args=(prompt, True))
    model_thread.start()

    model_thread_finished = False

    while not model_thread_finished:
        sleep(0.01)
        while not global_text.empty():
            new_embd = global_text.get()
            rkllm_output.append(new_embd)

        model_thread.join(timeout=0.005)
        model_thread_finished = not model_thread.is_alive()

    return rkllm_output

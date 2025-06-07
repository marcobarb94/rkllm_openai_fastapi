import logging
import threading
from time import sleep
from typing import Collection, Generator, List, Literal, Optional
import numpy as np
from core.util import EmbeddingsLLMData
from core.rkllm import global_text, global_state


def get_sentence_embedding(hidden_states: np.ndarray[float, float],
                           logits_aw: Optional[np.ndarray[float,
                                                          float]] = None,
                           emb_type: Optional[Literal["mean", "max", "last",
                                                      "weight"]] = None):
    """Quale metodo usare? ✔ Se vuoi semplicità → Mean pooling ✔ Se vuoi cogliere le parti dominanti → Max pooling ✔ Se il testo è breve e importante → Ultimo token ✔ Se vuoi massima qualità → Concatenazione o PCA
    """
    if emb_type is None:
        if logits_aw is not None:
            emb_type = "weight"
        else:
            emb_type = "mean"

    match emb_type:
        case "mean":
            _emb = np.mean(hidden_states, axis=0)
        case "max":
            _emb = np.amax(hidden_states, axis=0)
        case "last":
            _emb = hidden_states[-1, :]
        case "weight":
            logits = np.array(logits_aw)
            max_logits = np.max(logits, axis=1)
            attention_weights = softmax(max_logits)
            _emb = np.sum(hidden_states * attention_weights[:, np.newaxis],
                          axis=0)

    return _emb / np.linalg.norm(_emb)


def softmax(x):
    """
    Calcola la funzione softmax per una 1D array, utile per normalizzare i punteggi.
    """
    x_max = np.max(x)  # stabilizzazione numerica
    exps = np.exp(x - x_max)
    return exps / np.sum(exps)


def generate_embeddings(text: str, rkllm_model: ' core.rkllm.RKLLM', emb_type: str = "mean") -> np.array:
    global global_text, global_state

    prompt = text.strip()

    global_text._init(2)
    global_state = -1
    model_thread_hl = threading.Thread(target=rkllm_model.run,
                                       args=(prompt, "hidden_layer"))
    model_thread_hl.start()
    _emb_hidden: EmbeddingsLLMData = global_text.get()
    model_thread_hl.join(timeout=None)

    if False:
        global_text._init(2000)
        global_state = -1
        model_thread_logit = threading.Thread(target=rkllm_model.run,
                                            args=(prompt, "logit"))
        model_thread_logit.start()

        _emb_logit: List[EmbeddingsLLMData] = []

        model_thread_finished = False
        while not model_thread_finished:
            sleep(0.01)
            while not global_text.empty():
                new_embd = global_text.get()
                _emb_logit.append(new_embd)

            model_thread_logit.join(timeout=0.005)
            model_thread_finished = not model_thread_logit.is_alive()
    # _emb_logit: EmbeddingsLLMData = global_text.get()


    # _emb_hidden.merge(_emb_logit)

    return get_sentence_embedding(hidden_states=_emb_hidden.hidden_layer,
                                  logits_aw=None,#_emb_logit[0].logits,
                                  emb_type=emb_type)

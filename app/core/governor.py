from asyncio import sleep
import multiprocessing
from typing import AsyncGenerator, Tuple
from asyncio import Lock
from core.entities_llm import EngineComunication
from core.util import EmbeddingsLLMData, num_tokens_from_string


class Governor:
    lock: Lock
    cmd_queue: multiprocessing.Queue
    control_queue: multiprocessing.Queue
    result_queue: multiprocessing.Queue

    def __init__(self, cmd_queue: 'multiprocessing.Queue[EngineComunication]',
                 control_queue: multiprocessing.Queue,
                 result_queue: multiprocessing.Queue):
        self.lock = Lock()
        self.cmd_queue = cmd_queue
        self.control_queue = control_queue
        self.result_queue = result_queue

    async def stream_generate(self,
                              prompt: str) -> AsyncGenerator[str, str | int]:
        # bloccante fino a quando non sono finiti i job
        async with self.lock:
            try:
                self.cmd_queue.put(
                    EngineComunication(function_name="run",
                                       params={
                                           "infer_type": "generate",
                                           "prompt": prompt
                                       }))
                model_thread_finished = False
                self.result_queue._reset()
                completion_tokens = 0
                while not model_thread_finished:
                    await sleep(0.005)
                    while not self.result_queue.empty():
                        new_text = self.result_queue.get(timeout=10)
                        if new_text is None:
                            break
                        if type(new_text) is bool:
                            model_thread_finished = True
                            continue
                        completion_tokens += num_tokens_from_string(new_text)
                        yield new_text
                yield completion_tokens
            except GeneratorExit:
                self.cmd_queue.put(
                    EngineComunication(function_name="abort_job", params={}))
                raise

    async def hidden_layer(self, prompt: str) -> EmbeddingsLLMData:
        # bloccante fino a quando non sono finiti i job
        async with self.lock:
            self.cmd_queue.put(
                EngineComunication(function_name="run",
                                   params={
                                       "infer_type": "hidden_layer",
                                       "prompt": prompt
                                   }))
            self.result_queue._reset()
            model_thread_finished = False

            while not model_thread_finished:
                await sleep(0.005)
                while not self.result_queue.empty():
                    new_text = self.result_queue.get()
                    if type(new_text) is bool:
                        model_thread_finished = True
                        return EmbeddingsLLMData()
                    if type(new_text) is EmbeddingsLLMData:
                        return new_text

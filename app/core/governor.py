from asyncio import sleep, to_thread
import logging
from multiprocessing import Process
import multiprocessing
from queue import Empty
from typing import Any, AsyncGenerator, Dict, Optional, Tuple
from asyncio import Lock
from core.entities_llm import EngineComunication
from core.process import RKLLM_Engine
from core.util import EmbeddingsLLMData, num_tokens_from_string


class GovernorShell():
    engine: RKLLM_Engine
    process: Optional[Process]
    control_queue: multiprocessing.Queue
    cmd_queue: 'multiprocessing.Queue[EngineComunication]'
    engine_params: Dict[str, Any]

    def __init__(self, engine_params: Dict[str, Any],
                 cmd_queue: 'multiprocessing.Queue[EngineComunication]',
                 control_queue: multiprocessing.Queue):
        self.cmd_queue = cmd_queue
        self.control_queue = control_queue
        self.engine_params = engine_params
        self.process = None

    def start(self, engine_params: Optional[Dict[str, Any]] = None) -> None:
        if engine_params is None:
            engine_params = {}

        _dict_params = self.engine_params | engine_params

        self.engine = RKLLM_Engine(_dict_params,
                                   control_queue=self.control_queue,
                                   cmd_queue=self.cmd_queue)

        self.process = Process(target=self.engine.worker_func)
        self.process.start()

    async def shutdown(self) -> bool:
        self.cmd_queue.put("STOP")
        if self.is_on and self.process.is_alive():
            self.process.terminate()
            await sleep(3)
            #process.join(timeout=5)
            self.process.kill()
            return not self.process.is_alive()
        return True

    @property
    def is_on(self) -> bool:
        return self.process.is_alive() if self.process else False


class Governor:
    lock: Lock
    cmd_queue: multiprocessing.Queue
    control_queue: multiprocessing.Queue
    result_queue: multiprocessing.Queue
    governor_engine: GovernorShell

    def __init__(self, cmd_queue: 'multiprocessing.Queue[EngineComunication]',
                 control_queue: multiprocessing.Queue,
                 result_queue: multiprocessing.Queue,
                 governor_engine: GovernorShell):
        self.lock = Lock()
        self.cmd_queue = cmd_queue
        self.control_queue = control_queue
        self.result_queue = result_queue
        self.governor_engine = governor_engine

    async def stream_generate(
            self,
            prompt: str,
            enable_thinking: bool = False) -> AsyncGenerator[str, str | int]:
        # bloccante fino a quando non sono finiti i job

        if not self.governor_engine.is_on:
            self.governor_engine.start()
            await sleep(10)
            logging.info('engine start')
            
        logging.info('request lock')
        async with self.lock:
            logging.info('ok lock')
            try:
                self.cmd_queue.put(
                    EngineComunication(function_name="abort_job", params={}))
                self.result_queue._reset()
                while True:
                    try:
                        await to_thread(self.result_queue.get,timeout=1) # stop cross talk
                    except Empty:
                        break
                self.result_queue._reset()
                self.cmd_queue.put(
                    EngineComunication(function_name="run",
                                       params={
                                           "infer_type": "generate",
                                           "prompt": prompt,
                                           "enable_thinking": enable_thinking
                                       }))
                model_thread_finished = False
                completion_tokens = 0
                while not model_thread_finished and self.governor_engine.is_on:
                    await sleep(0.005)
                    while not self.result_queue.empty():
                        new_text = await to_thread(self.result_queue.get,timeout=25)
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
                    new_text = await to_thread(self.result_queue.get,timeout=60)
                    if type(new_text) is bool:
                        model_thread_finished = True
                        return EmbeddingsLLMData()
                    if type(new_text) is EmbeddingsLLMData:
                        return new_text

    async def stop(self) -> bool:
        return await self.governor_engine.shutdown()

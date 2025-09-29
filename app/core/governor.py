from asyncio import sleep, to_thread
import asyncio
import datetime
from datetime import timedelta
import logging
from multiprocessing import Process
import multiprocessing
from queue import Empty
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from asyncio import Lock
from core.entities_api import EngineParams
from core.entities_llm import EngineComunication, RKLLMPerfStat, UserdataCallback
from functools import lru_cache

from pydantic import BaseModel
from core.entities_llm import EngineComunication
from core.process import RKLLM_Engine
from core.util import EmbeddingsLLMData, QueueResult, num_tokens_from_string

from contextlib import asynccontextmanager


@asynccontextmanager
async def maybe_lock(lock, use_lock: bool):
    if use_lock:
        async with lock:
            yield
    else:
        yield


class GovernorShell():
    engine: RKLLM_Engine
    process: Optional[Process]
    control_queue: multiprocessing.Queue
    cmd_queue: 'multiprocessing.Queue[EngineComunication]'
    callback_queue: multiprocessing.Queue

    model_collection: List[EngineParams]
    _lock: asyncio.Lock
    model_running_id: int

    def __init__(self, model_collection: List[EngineParams],
                 cmd_queue: 'multiprocessing.Queue[EngineComunication]',
                 control_queue: multiprocessing.Queue,
                 callback_queue: multiprocessing.Queue):
        self.cmd_queue = cmd_queue
        self.control_queue = control_queue
        self.model_collection = model_collection
        self.callback_queue = callback_queue
        self.process = None
        self.shutdown_thread = None
        self._lock = asyncio.Lock()
        self.model_running_id = -1

    async def start(self, model_id: int = 0) -> bool:
        if model_id not in range(len(self.model_collection)):
            logging.error(f"Model_id {model_id} not found")
            return False

        async with self._lock:

            if self.is_on:
                if self.model_running_id == model_id:
                    return True
                await self.shutdown(ignore_lock=True)

            logging.info(
                f"Changing model from {self.model_running_id} to {model_id}")

            self.model_running_id = model_id
            _dict_params = self.model_collection[self.model_running_id]

            self.engine = RKLLM_Engine(_dict_params,
                                       control_queue=self.control_queue,
                                       cmd_queue=self.cmd_queue,
                                       callback_queue=self.callback_queue)

            self.process = Process(target=self.engine.worker_func)
            self.process.start()
            await sleep(10)
            logging.info(f'Start Engine: {self.process.is_alive()}')
            return True

    async def shutdown(self, ignore_lock: bool = False) -> bool:
        # Uso:
        async with maybe_lock(self._lock, not ignore_lock):
            self.cmd_queue.put("STOP")
            if self.is_on and self.process and self.process.is_alive():
                self.process.terminate()
                await sleep(3)
                #process.join(timeout=5)
                self.process.kill()
                logging.info(f'Stop Engine: {not self.process.is_alive()}')
                return not self.process.is_alive()
        return True

    @property
    def is_on(self) -> bool:
        return self.process.is_alive() if self.process else False

    @lru_cache(1)
    def get_model_catalog(self) -> Dict[str, Dict[str, int | bool]]:
        """Return all models name with the associated id

        :return: _description_
        :rtype: Dict[str, int]
        """
        _out = {}
        for i, _dict_params in enumerate(self.model_collection):
            for _name, _think in _dict_params.get_names():
                _out[_name] = {"id": i, "think": _think}

        return _out

    def get_model_id(self,
                     model_name: str) -> Tuple[int, bool, Dict[str, Any]]:
        if model_name in self.get_model_catalog():
            _re = self.get_model_catalog()[model_name]
            return (_re["id"], _re["think"],
                    self.model_collection[_re["id"]].tokenizer_config)
        raise ValueError(f"Model {model_name} not found!")


class Governor:
    lock: Lock
    cmd_queue: multiprocessing.Queue
    control_queue: multiprocessing.Queue
    result_queue: multiprocessing.Queue
    governor_engine: GovernorShell
    last_run: datetime.datetime

    def __init__(self, cmd_queue: 'multiprocessing.Queue[EngineComunication]',
                 control_queue: multiprocessing.Queue,
                 result_queue: multiprocessing.Queue,
                 governor_engine: GovernorShell):
        self.lock = Lock()
        self.cmd_queue = cmd_queue
        self.control_queue = control_queue
        self.result_queue = result_queue
        self.governor_engine = governor_engine
        self.last_run = datetime.datetime.now()

    async def stream_generate(self,
                              prompt: str,
                              model_name: str,
                              _id: Optional[int] = None
                              ) -> AsyncGenerator[QueueResult, QueueResult]:
        # bloccante fino a quando non sono finiti i job

        model_id, enable_thinking, _ = self.governor_engine.get_model_id(  # type: ignore
            model_name=model_name)

        logging.info(f'request lock: locked? {self.lock.locked()}')
        async with self.lock:
            logging.info('ok lock')
            await self.governor_engine.start(model_id)
            logging.info('engine start')

            _now = datetime.datetime.now()

            self.last_run = _now
            if _id is None:
                _id = int(_now.timestamp())

            try:
                self.cmd_queue.put(
                    EngineComunication(function_name="abort_job", params={}))
                self.result_queue._reset()
                while True:
                    try:
                        await to_thread(self.result_queue.get,
                                        timeout=2)  # stop cross talk
                    except Empty:
                        break
                self.result_queue._reset()
                self.cmd_queue.put(
                    EngineComunication(
                        function_name="run",
                        params={
                            "infer_type": "generate",
                            "prompt": prompt,
                            "enable_thinking": enable_thinking,
                            "userdata": UserdataCallback(id=_id)  #, queue_n=0)
                        }))
                model_thread_finished = False
                completion_tokens = 0
                # yield ""
                while not model_thread_finished and self.governor_engine.is_on:
                    await sleep(0.005)
                    while not self.result_queue.empty():
                        new_text: QueueResult = await to_thread(
                            self.result_queue.get, timeout=30)
                        if new_text is None:
                            break
                        if type(new_text.payload) is RKLLMPerfStat:
                            model_thread_finished = True
                            logging.info(new_text.payload)
                            yield new_text
                            break
                        # completion_tokens += num_tokens_from_string(new_text)
                        yield new_text
                yield completion_tokens
            except GeneratorExit:
                self.cmd_queue.put(
                    EngineComunication(function_name="abort_job", params={}))
                raise

    async def hidden_layer(self, prompt: str,
                           model_name: str) -> EmbeddingsLLMData:

        model_id, _, _ = self.governor_engine.get_model_id(  # type: ignore
            model_name=model_name)

        logging.info('request lock')
        async with self.lock:
            logging.info('ok lock')
            await self.governor_engine.start(model_id)
            logging.info('engine start')

            self.last_run = datetime.datetime.now()
            # bloccante fino a quando non sono finiti i job
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
                    new_text = await to_thread(self.result_queue.get,
                                               timeout=60)
                    if type(new_text) is bool:
                        model_thread_finished = True
                        return EmbeddingsLLMData()
                    if type(new_text) is EmbeddingsLLMData:
                        return new_text

    async def stop(self) -> bool:
        return await self.governor_engine.shutdown()

    async def schedule_shutdown(self, seconds: int = 600) -> None:
        while True:
            if (datetime.datetime.now() - self.last_run) > timedelta(
                    seconds=seconds) and self.governor_engine.is_on:
                logging.info(
                    f'Stopping Engine {(datetime.datetime.now() - self.last_run)}'
                )
                await self.stop()
            await sleep(seconds >> 1)

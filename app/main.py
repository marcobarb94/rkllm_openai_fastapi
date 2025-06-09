from asyncio import sleep
from contextlib import asynccontextmanager
import json
import multiprocessing
import resource
import threading
from fastapi import FastAPI
import os
from core.entities_api import AppConfig
from core.governor import Governor
from core.process import RKLLM_Engine
from core.entities_llm import LLMParams
from core.rkllm import control_queue, cmd_queue, result_queue
from api.router import router

resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("configs/config.json") as fp:
        config = AppConfig.model_validate(json.load(fp))
    if not os.path.exists(config.model_path):
        FileNotFoundError(f"Model not Found: {config.model_path}")

    # Load the ML model
    app.state.lock = threading.Lock()
    rkllm_model = RKLLM_Engine(
        {
            "model_path": config.model_path,
            "llm_params": config.llm_params
        },
        control_queue=control_queue,
        cmd_queue=cmd_queue)
    process = multiprocessing.Process(target=rkllm_model.worker_func)
    process.start()

    with open(config.path_tokenizer_config) as fp:
        app.state.tokenizer_config = json.load(fp)
    m_name = config.model_path.split("/")[-1].replace(".rkllm","")
    app.state.model_name = tuple(
        f"{m_name}-{th}"
        for th in ("std",
                   "reasoning")) if config.has_thinking else tuple(m_name)
    app.state.governor_engine = Governor(control_queue=control_queue,
                                         cmd_queue=cmd_queue,
                                         result_queue=result_queue)
    yield
    # Clean up the ML models and release the resources
    cmd_queue.put("STOP")
    await sleep(1000)
    process.terminate()
    process.join(timeout=5)
    process.kill()



app = FastAPI(lifespan=lifespan, title="RKLLM OpenAI", docs_url="/")
app.include_router(router)

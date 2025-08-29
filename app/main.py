from asyncio import sleep
import asyncio
from contextlib import asynccontextmanager
import contextlib
import json
import multiprocessing
import resource
import threading
from fastapi import FastAPI
import os
from core.entities_api import AppConfig
from core.governor import Governor, GovernorShell
from core.process import RKLLM_Engine
from core.entities_llm import LLMParams
from core.rkllm import control_queue, cmd_queue, result_queue
from api.router import router
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s")

resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("configs/config.json") as fp:
        config = AppConfig.model_validate(json.load(fp))
    if not os.path.exists(config.model_path):
        FileNotFoundError(f"Model not Found: {config.model_path}")

    # Load the ML model
    app.state.lock = asyncio.Lock()

    with open(config.path_tokenizer_config) as fp:
        app.state.tokenizer_config = json.load(fp)
    m_name = config.model_path.split("/")[-1].replace(".rkllm", "")
    app.state.model_name = tuple(
        f"{th}-{m_name}"
        for th in ("std", "think")) if config.has_thinking else tuple(m_name)

    gov_engine = GovernorShell(engine_params={
        "model_path": config.model_path,
        "llm_params": config.llm_params
    },
                               control_queue=control_queue,
                               cmd_queue=cmd_queue)
    # gov_engine.start()
    app.state.governor_engine = Governor(control_queue=control_queue,
                                         cmd_queue=cmd_queue,
                                         result_queue=result_queue,
                                         governor_engine=gov_engine)
    yield
    logging.info("Stop")
    # Clean up the ML models and release the resources

    logging.info("-")
    print("Shutting down")
    await app.state.governor_engine.stop()
    await sleep(3)
    logging.info("Terminate")
    with contextlib.suppress(Exception):
        result_queue.close()
        cmd_queue.close()


app = FastAPI(lifespan=lifespan, title="RKLLM OpenAI", docs_url="/")
app.include_router(router)

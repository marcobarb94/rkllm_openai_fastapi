from asyncio import sleep
from contextlib import asynccontextmanager
import json
import multiprocessing
import resource
import threading
from fastapi import FastAPI
import os
from core.governor import Governor
from core.process import RKLLM_Engine
from core.entities_llm import LLMParams
from core.rkllm import RKLLM, control_queue, cmd_queue, result_queue
from api.router import router

resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("configs/config.json") as fp:
        config = json.load(fp)
    if not os.path.exists(config["model_path"]):
        FileNotFoundError(f"Model not Found: {config['model_path']}")

    if "llm_params" in config:
        llm_params = LLMParams(**config['llm_params'])

    else:
        llm_params = LLMParams()

    # Load the ML model
    app.state.lock = threading.Lock()
    rkllm_model = RKLLM_Engine(
        {
            "model_path": config["model_path"],
            "llm_params": llm_params
        },
        control_queue=control_queue,
        cmd_queue=cmd_queue)
    process = multiprocessing.Process(target=rkllm_model.worker_func)
    process.start()

    with open(config["path_tokenizer_config"]) as fp:
        app.state.tokenizer_config = json.load(fp)
    app.state.model_name = config["model_path"].split("/")[-1]
    app.state.governor_engine = Governor(control_queue=control_queue,
                                         cmd_queue=cmd_queue,
                                         result_queue=result_queue)
    yield
    # Clean up the ML models and release the resources
    cmd_queue.put("STOP")
    await sleep(1000)
    process.terminate()
    process.join()


app = FastAPI(lifespan=lifespan, title="RKLLM OpenAI", docs_url="/")
app.include_router(router)

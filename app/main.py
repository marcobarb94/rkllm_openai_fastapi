from asyncio import sleep
import asyncio
from contextlib import asynccontextmanager
import contextlib
import json
import resource
from fastapi import FastAPI
import os
from core.entities_api import AppConfig
from core.governor import Governor, GovernorShell
from core.rkllm import control_queue, cmd_queue, result_queue, callback_queue
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


    # Load the ML model
    app.state.lock = asyncio.Lock()

    gov_engine = GovernorShell(model_collection=config.model_collection,
                               control_queue=control_queue,
                               cmd_queue=cmd_queue,callback_queue=callback_queue)
    # gov_engine.start()
    app.state.governor_engine = Governor(control_queue=control_queue,
                                         cmd_queue=cmd_queue,
                                         result_queue=result_queue,
                                         governor_engine=gov_engine)
    task_stop = asyncio.create_task(
        app.state.governor_engine.schedule_shutdown(seconds=1000))
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

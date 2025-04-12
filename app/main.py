from contextlib import asynccontextmanager
import json
import resource
import threading
from fastapi import FastAPI

from app.core.rkllm import RKLLM


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("configs/config.json") as fp:
        config = json.load(fp)
    # Load the ML model
    app.state.rkllm_model = RKLLM(config["model_path"],
                                  config["target_platform"])
    yield
    # Clean up the ML models and release the resources
    app.state.rkllm_model.release()


if __name__ == "__main__":
    global app
    app = FastAPI(lifespan=lifespan, title="RKLLM OpenAI", docs_url="/")

    app.state.lock = threading.Lock()
    # Установка ограничения на количество файловых дескрипторов
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

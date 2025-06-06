from contextlib import asynccontextmanager
import json
import resource
import threading
from fastapi import FastAPI
import os
from core.entities_llm import LLMParams
from core.rkllm import RKLLM
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
    app.state.rkllm_model = RKLLM(config["model_path"],
                                  config["target_platform"],
                                  llm_params=llm_params)
    with open(config["path_tokenizer_config"]) as fp:
        app.state.tokenizer_config = json.load(fp)
    app.state.model_name = config["model_path"].split("/")[-1]
    yield
    # Clean up the ML models and release the resources
    app.state.rkllm_model.release()


app = FastAPI(lifespan=lifespan, title="RKLLM OpenAI", docs_url="/")
app.include_router(router)

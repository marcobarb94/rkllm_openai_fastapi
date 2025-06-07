import asyncio
import json
import logging
import threading
from time import time, sleep
from typing import *
from fastapi import APIRouter, FastAPI, Request, Response, status
from fastapi.responses import StreamingResponse, JSONResponse

from core.embeddings import generate_embeddings
from core.entities_api import ChatCompletionChunk, ChatCompletionRequest, CompletionRequest, ChatCompletionResponse, CompletionResponse, EmbeddingData, EmbeddingRequest, EmbeddingResponse, Model, ModelsResponse, OpenAIErrorDetail, OpenAIErrorResponse, OpenAIRoles
from core.util import num_tokens_from_string, parse_message_to_prompt
from core.rkllm import global_text, global_state

router = APIRouter(prefix="/v1")


@router.post('/chat/completions', response_model=None)
def chat_completions(
    data: ChatCompletionRequest, request: Request, response: Response
) -> StreamingResponse | ChatCompletionResponse | OpenAIErrorResponse:
    global global_text, global_state

    if request.app.state.lock.locked():
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return OpenAIErrorResponse(error=OpenAIErrorDetail(
            message="Server RKLLM is busy! Please try again later.",
            type="server_error"))

    with request.app.state.lock:
        try:

            global_text._init(4096)
            global_state = -1

            stream = data.stream
            model = data.model if data.model else request.app.state.model_name

            prompt = parse_message_to_prompt(
                data.messages, request.app.state.tokenizer_config)

            prompt = prompt.strip()

            def generate() -> Generator:
                nonlocal prompt
                rkllm_output = ""
                prompt_tokens = num_tokens_from_string(prompt)
                completion_tokens = 0

                model_thread = threading.Thread(
                    target=request.app.state.rkllm_model.run, args=(prompt, ))
                model_thread.start()

                model_thread_finished = False

                while not model_thread_finished:
                    sleep(0.005)
                    while not global_text.empty():
                        new_text = global_text.get()
                        rkllm_output += new_text
                        completion_tokens += num_tokens_from_string(new_text)

                        if stream:
                            _res = ChatCompletionChunk(
                                id=f"chatcmpl-{time()}",
                                object="chat.completion.chunk",
                                created=int(time()),
                                model=model,
                                choices=[{
                                    "index": 0,
                                    "delta": {
                                        "content": new_text
                                    },
                                    "finish_reason": None
                                }])

                            yield f"data: {_res.model_dump_json()}\n\n"
                        sleep(0.005)

                    model_thread.join(timeout=0.005)
                    model_thread_finished = not model_thread.is_alive()
                    if request._is_disconnected:  # await request.is_disconnected():
                        logging.info("User Stops")

                if stream:
                    final_response = ChatCompletionChunk(
                        **{
                            "id":
                            f"chatcmpl-{time()}",
                            "object":
                            "chat.completion.chunk",
                            "created":
                            int(time()),
                            "model":
                            model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        })
                    yield f"data: {final_response.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                else:
                    yield ChatCompletionResponse(
                        **{
                            "id":
                            f"chatcmpl-{time()}",
                            "object":
                            "chat.completion",
                            "created":
                            int(time()),
                            "model":
                            model,
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": rkllm_output,
                                },
                                "finish_reason": "stop",
                            }],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens +
                                completion_tokens,
                            },
                        })

            if stream:
                return StreamingResponse(generate(),
                                         media_type='text/event-stream')
            else:
                return next(generate())
        except Exception as e:
            logging.error(e)
            return OpenAIErrorResponse(
                error=OpenAIErrorDetail(message=str(e), type="server_error"))


# @router.post('/completions')
# def completions(
#         data: CompletionRequest, request: Request,
#         response: Response) -> CompletionResponse | OpenAIErrorResponse:
#     return {}


@router.get('/models')
def get_models(request: Request) -> ModelsResponse:
    return ModelsResponse(
        object="list",
        data=[Model(id=request.app.state.model_name, object="model")])


@router.post("/embeddings",
             response_model=EmbeddingResponse | OpenAIErrorResponse)
async def get_embedding(request: Request, ebm_request: EmbeddingRequest):
    with request.app.state.lock:
        # Creiamo la lista di embeddings con indice
        try:
            embedding_list = [
                EmbeddingData(object="embedding",
                              embedding=await asyncio.to_thread(
                                  generate_embeddings,
                                  text=_r,
                                  rkllm_model=request.app.state.rkllm_model),
                              index=i)
                for i, _r in enumerate(ebm_request.input)
            ]
        except Exception as e:
            logging.error(e)
            return OpenAIErrorResponse(
                error=OpenAIErrorDetail(message=str(e), type="server_error"))

    return EmbeddingResponse(
        object="list",
        data=embedding_list,
        model=ebm_request.model,
        usage={
            "total_tokens":
            sum(len(text.split()) for text in ebm_request.input)
        })

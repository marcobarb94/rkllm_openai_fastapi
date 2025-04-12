import json
import logging
import threading
import time
from typing import *
from fastapi import APIRouter, FastAPI, Request, Response, status
from fastapi.responses import StreamingResponse, JSONResponse

from app.core.entities_api import ChatCompletionChunk, ChatCompletionRequest, CompletionRequest, ChatCompletionResponse, CompletionResponse, Model, ModelsResponse, OpenAIErrorDetail, OpenAIErrorResponse, OpenAIRoles
from core.util import num_tokens_from_string, parse_message_to_prompt
from transformers import AutoTokenizer, AutoProcessor

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

            global_text = []
            global_state = -1

            stream = data.stream
            model = data.model

            prompt = parse_message_to_prompt(
                data.messages, request.app.state.tokenizer_config)

            prompt = prompt.strip()

            def generate():
                nonlocal prompt
                rkllm_output = ""
                prompt_tokens = num_tokens_from_string(prompt)
                completion_tokens = 0

                model_thread = threading.Thread(
                    target=request.app.state.rkllm_model.run, args=(prompt, ))
                model_thread.start()

                model_thread_finished = False

                while not model_thread_finished:
                    while len(global_text) > 0:
                        new_text = global_text.pop(0)
                        rkllm_output += new_text
                        completion_tokens += num_tokens_from_string(new_text)

                        if stream:
                            response = ChatCompletionChunk(
                                **{
                                    "id":
                                    f"chatcmpl-{time.time()}",
                                    "object":
                                    "chat.completion.chunk",
                                    "created":
                                    int(time.time()),
                                    "model":
                                    model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {
                                            "content": new_text
                                        },
                                        "finish_reason": None
                                    }]
                                })

                            yield f"data: {json.dumps(response)}\n\n"
                        time.sleep(0.005)

                    model_thread.join(timeout=0.005)
                    model_thread_finished = not model_thread.is_alive()

                if stream:
                    final_response = ChatCompletionChunk(
                        **{
                            "id":
                            f"chatcmpl-{time.time()}",
                            "object":
                            "chat.completion.chunk",
                            "created":
                            int(time.time()),
                            "model":
                            model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        })
                    yield f"data: {json.dumps(final_response)}\n\n"
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
                error=OpenAIErrorDetail(message=e, type="server_error"))


# @router.post('/completions')
# def completions(
#         data: CompletionRequest, request: Request,
#         response: Response) -> CompletionResponse | OpenAIErrorResponse:
#     return {}


@router.post('/models')
def get_models(request: Request) -> ModelsResponse:
    return ModelsResponse(object="list",
        data=[Model(id=request.app.state.model_name, object="model")])

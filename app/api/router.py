import asyncio
import json
import logging
import threading
from time import time, sleep
from typing import *
from fastapi import APIRouter, FastAPI, Request, Response, status
from fastapi.responses import StreamingResponse, JSONResponse

from core.embeddings import generate_embeddings, get_sentence_embedding
from core.entities_api import ChatCompletionChunk, ChatCompletionRequest, CompletionChoice, CompletionRequest, ChatCompletionResponse, CompletionResponse, EmbeddingData, EmbeddingRequest, EmbeddingResponse, Model, ModelsResponse, OpenAIErrorDetail, OpenAIErrorResponse, OpenAIRoles
from core.governor import Governor
from core.util import num_tokens_from_string, parse_message_to_prompt
from core.rkllm import result_queue, global_state

router = APIRouter(prefix="/v1")


@router.post('/chat/completions', response_model=None)
async def chat_completions(
    data: ChatCompletionRequest, request: Request, response: Response
) -> StreamingResponse | ChatCompletionResponse | OpenAIErrorResponse:
    global result_queue, global_state

    if False:  # request.app.state.lock.locked():
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return OpenAIErrorResponse(error=OpenAIErrorDetail(
            message="Server RKLLM is busy! Please try again later.",
            type="server_error"))
    
    logging.info(f"New Request! Locked: {request.app.state.lock.locked()}")

    async with request.app.state.lock:
        try:

            stream = data.stream
            model = data.model if data.model else request.app.state.model_name[
                0]

            prompt = parse_message_to_prompt(
                data.messages,
                request.app.state.tokenizer_config,
                enable_thinking=(enable_thinking :=
                                 model.endswith("reasoning")))

            prompt = prompt.strip()

            async def generate(prompt: str, gov: Governor) -> AsyncGenerator:

                prompt_tokens = num_tokens_from_string(prompt)
                completion_tokens = 0
                rkllm_output = ""

                async for new_text in gov.stream_generate(
                        prompt, enable_thinking=enable_thinking):
                    if type(new_text) is int:
                        completion_tokens = new_text
                        continue

                    rkllm_output += new_text

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

                    if await request.is_disconnected(
                    ):  # await request.is_disconnected():
                        break

                if stream:
                    final_response = ChatCompletionChunk(
                        id=f"chatcmpl-{time()}",
                        object="chat.completion.chunk",
                        created=int(time()),
                        model=model,
                        choices=[{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }])
                    # logging.info(rkllm_output) # TODO: REMOVE
                    yield f"data: {final_response.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                else:
                    yield ChatCompletionResponse(
                        id=f"chatcmpl-{time()}",
                        object="chat.completion",
                        created=int(time()),
                        model=model,
                        choices=[{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": rkllm_output,
                            },
                            "finish_reason": "stop",
                        }],
                        usage={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    )

            if stream:
                return StreamingResponse(generate(
                    prompt, request.app.state.governor_engine),
                                         media_type='text/event-stream')
            else:
                return await anext(
                    generate(prompt, request.app.state.governor_engine))
        except Exception as e:
            logging.error(e)
            return OpenAIErrorResponse(
                error=OpenAIErrorDetail(message=str(e), type="server_error"))


@router.post('/completions', response_model=None)
async def completions(
    data: CompletionRequest, request: Request, response: Response
) -> StreamingResponse | CompletionResponse | OpenAIErrorResponse:
    with request.app.state.lock:
        try:

            stream = data.stream
            model = data.model if data.model else request.app.state.model_name

            prompt = data.prompt

            prompt = prompt.strip()

            async def generate(prompt: str, gov: Governor) -> AsyncGenerator:

                prompt_tokens = num_tokens_from_string(prompt)
                completion_tokens = 0
                rkllm_output = ""
                _in = 0
                _response_id = f"cmpl-{time()}"

                async for new_text in gov.stream_generate(prompt):
                    if type(new_text) is int:
                        completion_tokens = new_text
                        continue

                    rkllm_output += new_text

                    if stream:
                        _in += 1
                        _res = CompletionResponse(id=_response_id,
                                                  created=int(time()),
                                                  model=model,
                                                  choices=[
                                                      CompletionChoice(
                                                          index=_in,
                                                          text=new_text)
                                                  ])

                        yield f"data: {_res.model_dump_json()}\n\n"

                    if await request.is_disconnected(
                    ):  # await request.is_disconnected():
                        break

                if stream:
                    final_response = CompletionResponse(id=_response_id,
                                                        created=int(time()),
                                                        model=model,
                                                        choices=[{
                                                            "index":
                                                            0,
                                                            "finish_reason":
                                                            "stop",
                                                            "text":
                                                            ""
                                                        }])
                    yield f"data: {final_response.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                else:
                    yield CompletionResponse(
                        id=_response_id,
                        created=int(time()),
                        model=model,
                        choices=[{
                            "index": 0,
                            "text": rkllm_output,
                            "finish_reason": "stop",
                        }],
                        usage={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    )

            if stream:
                return StreamingResponse(generate(
                    prompt, request.app.state.governor_engine),
                                         media_type='text/event-stream')
            else:
                return await anext(
                    generate(prompt, request.app.state.governor_engine))
        except Exception as e:
            logging.error(e)
            return OpenAIErrorResponse(
                error=OpenAIErrorDetail(message=str(e), type="server_error"))


@router.get('/models')
async def get_models(request: Request) -> ModelsResponse:
    return ModelsResponse(object="list",
                          data=[
                              Model(id=m_mod, object="model")
                              for m_mod in request.app.state.model_name
                          ])


@router.post("/embeddings",
             response_model=EmbeddingResponse | OpenAIErrorResponse)
async def get_embedding(request: Request, ebm_request: EmbeddingRequest):
    with request.app.state.lock:
        # Creiamo la lista di embeddings con indice
        try:
            embedding_list = [
                EmbeddingData(
                    object="embedding",
                    embedding=get_sentence_embedding(
                        hidden_states=(await request.app.state.governor_engine.
                                       hidden_layer(_r)).hidden_layer,
                        logits_aw=None,  #_emb_logit[0].logits,
                        emb_type="mean"),
                    index=i) for i, _r in enumerate(ebm_request.input)
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

@router.get("/is_busy")
async def is_busy(request: Request) -> bool:
    return request.app.state.lock.locked()
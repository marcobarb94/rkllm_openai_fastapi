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
from core.entities_llm import RKLLMPerfStat
from core.governor import Governor
from core.util import QueueResult, num_tokens_from_string, parse_message_to_prompt
from core.rkllm import result_queue, global_state

router = APIRouter(prefix="/v1")


def init_funct(data: ChatCompletionRequest | CompletionRequest,
               ge: Governor,
               completions_only: bool = False) -> Tuple[str, bool, str]:

    stream = data.stream
    try:
        _, enable_thinking, tokenizer_config = ge.governor_engine.get_model_id(
            model_name=data.model)
    except ValueError as e:
        logging.error(e)
        return OpenAIErrorResponse(error=OpenAIErrorDetail(
            message=str(e), type="invalid_request_error"))

    prompt = (
        "\n".join(data.messages) if isinstance(data.messages, list) else
        data.messages) if completions_only else parse_message_to_prompt(
            data.messages, tokenizer_config, enable_thinking=enable_thinking)

    prompt = prompt.strip()
    return prompt, stream, data.model


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

    logging.debug(f"New Request! Locked: {request.app.state.lock.locked()}")
    # TODO: studio su batch inference
    ge: Governor = request.app.state.governor_engine
    prompt, stream, model_name = init_funct(data, ge)

    logging.info(f"Request for {model_name}")

    async def generate(prompt: str, gov: Governor) -> AsyncGenerator:

        prompt_tokens = num_tokens_from_string(prompt)
        completion_tokens = 0
        rkllm_output = ""
        usage_info: RKLLMPerfStat = None
        _id = int(time())

        async for q_res in gov.stream_generate(prompt,
                                               model_name=model_name,
                                               _id=_id):
            if type(q_res) is not QueueResult:
                continue
            if q_res.queue_id != _id:
                logging.info(f"{q_res.queue_id} != {_id}")
                continue

            _out = q_res.payload
            _type = type(_out)
            if _type is int:
                completion_tokens = _out
                continue
            elif _type is RKLLMPerfStat:
                usage_info = _out
                continue

            rkllm_output += _out

            if stream:
                _res = ChatCompletionChunk(id=f"chatcmpl-{time()}",
                                           object="chat.completion.chunk",
                                           created=int(time()),
                                           model=model_name,
                                           choices=[{
                                               "index": 0,
                                               "delta": {
                                                   "content": _out
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
                model=model_name,
                choices=[{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }],
                usage=usage_info.openai_usage(),
            )
            # logging.info(rkllm_output) # TODO: REMOVE
            yield f"data: {final_response.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        else:
            yield ChatCompletionResponse(
                id=f"chatcmpl-{time()}",
                object="chat.completion",
                created=int(time()),
                model=model_name,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": rkllm_output,
                    },
                    "finish_reason": "stop",
                }],
                usage=usage_info.openai_usage(),
            )

    async with request.app.state.lock:
        try:
            if stream:
                return StreamingResponse(generate(prompt, ge),
                                         media_type='text/event-stream')
            else:
                return await anext(generate(prompt, ge))
        except Exception as e:
            logging.error(e)
            return OpenAIErrorResponse(
                error=OpenAIErrorDetail(message=str(e), type="server_error"))


@router.post('/completions', response_model=None)
async def completions(
    data: CompletionRequest, request: Request, response: Response
) -> StreamingResponse | CompletionResponse | OpenAIErrorResponse:

    logging.info(f"New Request! Locked: {request.app.state.lock.locked()}")
    ge: Governor = request.app.state.governor_engine
    prompt, stream, model_name = init_funct(data, ge, completions_only=True)

    async with request.app.state.lock:
        try:

            async def generate(prompt: str, gov: Governor) -> AsyncGenerator:

                prompt_tokens = num_tokens_from_string(prompt)
                completion_tokens = 0
                rkllm_output = ""
                _in = 0
                _response_id = f"cmpl-{time()}"
                _id = int(time())
                usage_info: RKLLMPerfStat = None

                async for q_res in gov.stream_generate(prompt,
                                                       model_name=model_name,
                                                       _id=_id):
                    if q_res.queue_id != _id:
                        logging.info(f"{q_res.queue_id} != {_id}")
                        continue

                    _out = q_res.payload
                    _type = type(_out)
                    if _type is int:
                        completion_tokens = _out
                        continue
                    elif _type is RKLLMPerfStat:
                        usage_info = _out
                        continue

                    rkllm_output += _out

                    if stream:
                        _in += 1
                        _res = CompletionResponse(
                            id=_response_id,
                            created=int(time()),
                            model=model_name,
                            choices=[CompletionChoice(index=_in, text=_out)])

                        yield f"data: {_res.model_dump_json()}\n\n"

                    if await request.is_disconnected(
                    ):  # await request.is_disconnected():
                        break

                if stream:
                    final_response = CompletionResponse(
                        id=_response_id,
                        created=int(time()),
                        model=model_name,
                        choices=[{
                            "index": 0,
                            "finish_reason": "stop",
                            "text": ""
                        }],
                        usage=usage_info.openai_usage())
                    yield f"data: {final_response.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                else:
                    yield CompletionResponse(id=_response_id,
                                             created=int(time()),
                                             model=model_name,
                                             choices=[{
                                                 "index": 0,
                                                 "text": rkllm_output,
                                                 "finish_reason": "stop",
                                             }],
                                             usage=usage_info.openai_usage())

            if stream:
                return StreamingResponse(generate(prompt, ge),
                                         media_type='text/event-stream')
            else:
                return await anext(generate(prompt, ge))
        except Exception as e:
            logging.error(e)
            return OpenAIErrorResponse(
                error=OpenAIErrorDetail(message=str(e), type="server_error"))


@router.get('/models')
async def get_models(request: Request) -> ModelsResponse:
    ge: Governor = request.app.state.governor_engine
    return ModelsResponse(
        object="list",
        data=[
            Model(id=m_mod, object="model")
            for m_mod in ge.governor_engine.get_model_catalog()
        ])


@router.post("/embeddings",
             response_model=EmbeddingResponse | OpenAIErrorResponse)
async def get_embedding(request: Request, ebm_request: EmbeddingRequest):
    ge: Governor = request.app.state.governor_engine

    with request.app.state.lock:
        # Creiamo la lista di embeddings con indice
        try:
            embedding_list = [
                EmbeddingData(
                    object="embedding",
                    embedding=get_sentence_embedding(
                        hidden_states=(await ge.hidden_layer(
                            _r, ebm_request.model)).hidden_layer,
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


@router.put('/stop_engine')
async def stop_engine(request: Request) -> bool:
    ge: Governor = request.app.state.governor_engine
    return await ge.stop()

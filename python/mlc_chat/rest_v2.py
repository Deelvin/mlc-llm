import argparse
import asyncio
import uuid
import os
import subprocess
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from dataclasses import dataclass, field, fields
from typing import Optional

from .base import set_global_random_seed
# from .chat_module import ChatModule
from .interface.openai_api import *

import numpy as np

from .async_llm_engine import AsyncLLMEngine
from .async_sampling_params import SamplingParams
from .async_outputs import RequestOutput
from .async_arg_utils import EngineArgs

@dataclass
class RestAPIArgs:
    """RestAPIArgs is the dataclass that organizes the arguments used for starting a REST API server."""
    host: str = field(
        default="127.0.0.1",
        metadata={
            "help": (
                """
                The host at which the server should be started, defaults to ``127.0.0.1``.
                """
            )
        }
    )
    port: int = field(
        default=8000,
        metadata={
            "help": (
                """
                The port on which the server should be started, defaults to ``8000``.
                """
            )
        }
    )

def convert_args_to_argparser() -> argparse.ArgumentParser:
    """Convert from RestAPIArgs to an equivalent ArgumentParser."""
    args = argparse.ArgumentParser("MLC Chat REST API")
    for field in fields(RestAPIArgs):
        name = field.name.replace("_", "-")
        field_name = f"--{name}"
        # `kwargs` contains `help`, `choices`, and `action`
        kwargs = field.metadata.copy()
        if field.type == bool:
            # boolean arguments do not need to specify `type`
            args.add_argument(field_name, default=field.default, **kwargs)
        else:
            args.add_argument(field_name, type=field.type, default=field.default, **kwargs)
    return args


engine = None



@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    if ARGS.random_seed is not None:
        set_global_random_seed(ARGS.random_seed)
    engine_args = EngineArgs.from_cli_args(ARGS)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    yield

    engine = None


app = FastAPI(lifespan=lifespan)

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/chat/completions")
async def request_completion(request: ChatCompletionRequest):
    """
    Creates model response for the given chat conversation.
    """

    global engine
    def random_uuid() -> str:
        return str(uuid.uuid4().hex)

    # TODO(amalyshe) remove this verification and handle a case properly
    if len(request.messages) > 1:
            raise ValueError(
                """
                The /v1/chat/completions endpoint currently only supports single message prompts.
                Please ensure your request contains only one message
                """)

    if request.stream:
        # TODO(amalyshe): handle streamed requests
        raise ValueError(
                """
                Streamsed requests are not supported yet
                """)
    else:
        request_id = f"cmpl-{random_uuid()}"
        created_time = int(time.time())
        model_name = request.model
        try:
            # TODO(amalyshe): since sampling params are disabled yet in mlc-llm ChatCompletionRequest
            # we are ignoring their initialization, need to enable
            sampling_params = SamplingParams(
                # n=request.n,
                # presence_penalty=request.presence_penalty,
                # frequency_penalty=request.frequency_penalty,
                # temperature=request.temperature,
                # top_p=request.top_p,
                # stop=request.stop,
                # max_tokens=request.max_tokens,
                # best_of=request.best_of,
                # top_k=request.top_k,
                # ignore_eos=request.ignore_eos,
                # use_beam_search=request.use_beam_search,
            )
        except ValueError as e:
            raise ValueError( """
                issues with sampling parameters
                """)

        result_generator = engine.generate(request.messages[0].content, sampling_params, request_id)

        # Non-streaming response
        final_res: RequestOutput = None
        async for res in result_generator:
            # TODO(amalyshe): fix a branch for disconnected request
            # if await raw_request.is_disconnected():
            #     # Abort the request if the client disconnects.
            #     await abort_request()
            #     return create_error_response(HTTPStatus.BAD_REQUEST,
            #                                 "Client disconnected")
            final_res = res
        assert final_res is not None
        choices = []
        for output in final_res.outputs:
            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=ChatMessage(role="assistant", content=output.text),
                finish_reason=output.finish_reason,
            )
            choices.append(choice_data)

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        # TODO(amalyshe): figure out how we can appear in this place if there is another verificatio on streaming above
        # if request.stream:
        #     # When user requests streaming but we don't stream, we still need to
        #     # return a streaming response with a single event.
        #     response_json = response.json(ensure_ascii=False)

        #     async def fake_stream_generator() -> AsyncGenerator[str, None]:
        #         yield f"data: {response_json}\n\n"
        #         yield "data: [DONE]\n\n"

        #     return StreamingResponse(fake_stream_generator(),
        #                             media_type="text/event-stream")

        return response

@app.post("/v1/completions")
async def request_completion(request: CompletionRequest):
    """
    Creates a completion for a given prompt.
    """
    session["chat_mod"].reset_chat()
    # Langchain's load_qa_chain.run expects the input to be a list with the query
    if isinstance(request.prompt, list):
        if len(request.prompt) > 1:
            raise ValueError(
                """
                The /v1/completions endpoint currently only supports single message prompts.
                Please ensure your request contains only one message
                """)
        prompt = request.prompt[0]
    else:
        prompt = request.prompt

    msg = session["chat_mod"].generate(prompt=prompt)

    return CompletionResponse(
        choices=[CompletionResponseChoice(index=0, text=msg)],
        # TODO: Fill in correct usage info
        usage=UsageInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )


@app.post("/v1/embeddings")
async def request_embeddings(request: EmbeddingsRequest):
    """
    Gets embedding for some text.
    """
    inps = []
    if type(request.input) == str:
        inps.append(request.input)
    elif type(request.input) == list:
        inps = request.input
    else:
        assert f"Invalid input type {type(request.input)}"
    
    data = []
    for i, inp in enumerate(inps):
        session["chat_mod"].reset_chat()
        emb = session["chat_mod"].embed_text(input=inp).numpy()
        mean_emb = np.squeeze(np.mean(emb, axis=1), axis=0)
        norm_emb = mean_emb / np.linalg.norm(mean_emb)
        data.append({"object": "embedding", "embedding": norm_emb.tolist(), "index": i})
    # TODO: Fill in correct usage info
    return EmbeddingsResponse(
        data=data,
        usage=UsageInfo(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )
    )


@app.post("/chat/reset")
async def reset():
    """
    Reset the chat for the currently initialized model.
    """
    session["chat_mod"].reset_chat()


@app.get("/stats")
async def read_stats():
    """
    Get the runtime stats.
    """
    return session["chat_mod"].stats()


ARGS = convert_args_to_argparser()
ARGS = EngineArgs.add_cli_args(ARGS)
ARGS = ARGS.parse_args()
if __name__ == "__main__":
    uvicorn.run("mlc_chat.rest_v2:app", host=ARGS.host, port=ARGS.port, reload=False, access_log=False)

from mlc_serve.engine import (
    Request,
    ChatMessage,
    DebugOptions,
    SamplingParams,
    StoppingCriteria,
    FinishReason,
    get_engine_config,
)
from mlc_serve.engine.staging_engine import StagingInferenceEngine
from mlc_serve.engine.sync_engine import SynchronousInferenceEngine
from mlc_serve.model.base import get_model_artifact_config
from mlc_serve.model.paged_cache_model import HfTokenizerModule, PagedCacheModelModule
from mlc_serve.utils import get_default_mlc_serve_argparser, postproc_mlc_serve_args
import random


def create_engine(
    model_artifact_path,
    max_num_batched_tokens,
    use_staging_engine,
):
    engine_config = get_engine_config(
        {
            "use_staging_engine": use_staging_engine,
            "max_num_batched_tokens": max_num_batched_tokens,
            # Use defaults for "min_decode_steps", "max_decode_steps"
        }
    )

    if use_staging_engine:
        engine = StagingInferenceEngine(
            tokenizer_module=HfTokenizerModule(model_artifact_path),
            model_module_loader=PagedCacheModelModule,
            model_module_loader_kwargs={
                "model_artifact_path": model_artifact_path,
                "engine_config": engine_config,
            },
        )
        engine.start()
    else:
        engine = SynchronousInferenceEngine(
            PagedCacheModelModule(
                model_artifact_path=model_artifact_path,
                engine_config=engine_config,
            )
        )
    return engine


def create_request(
    idx,
    prompt,
    temp,
    freq_pen,
    pre_pen,
    max_tokens,
    stop,
    ignore_eos,
    top_logprobs=0,
    logprobs=False,
    logit_bias=None,
):
    return Request(
        request_id=str(idx),
        messages=[ChatMessage(role="user", content=prompt)],
        sampling_params=SamplingParams(
            temperature=temp,
            frequency_penalty=freq_pen,
            presence_penalty=pre_pen,
            logit_bias=logit_bias,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        ),
        stopping_criteria=StoppingCriteria(max_tokens=max_tokens, stop_sequences=stop),
        debug_options=DebugOptions(ignore_eos=ignore_eos),
    )


def _test_max_tokens(
    engine,
    num_requests=5,
    ignore_eos=False,
):
    prompt = "Write a merge sort program in Python."
    requests = [
        create_request(
            idx=str(n - 1),
            prompt=prompt,
            temp=0,
            freq_pen=0,
            pre_pen=0,
            max_tokens=n,
            stop=None,
            ignore_eos=ignore_eos,
        )
        for n in range(1, num_requests)
    ]
    engine.add(requests)

    generated = ["" for _ in range(num_requests)]

    while engine.has_pending_requests():
        results = engine.step()
        for res in results.outputs:
            assert len(res.sequences) == 1
            seq = res.sequences[0]

            if seq.is_finished:
                assert (
                    seq.num_generated_tokens
                    == requests[int(res.request_id)].stopping_criteria.max_tokens
                )
                assert seq.finish_reason == FinishReason.Length
            else:
                generated[int(res.request_id)] += seq.delta


def _test_max_context_length(
    model_artifact_path,
    engine,
    num_requests=5,
    ignore_eos=False,
):
    model_artifact_config = get_model_artifact_config(model_artifact_path)
    max_context_length = model_artifact_config.max_context_length
    prompt = "hi " * (max_context_length - 15)

    requests = [
        create_request(
            idx=str(n),
            prompt=prompt,
            temp=0,
            freq_pen=0,
            pre_pen=0,
            max_tokens=None,
            stop=None,
            ignore_eos=ignore_eos,
        )
        for n in range(num_requests)
    ]
    engine.add(requests)

    generated = ["" for _ in range(num_requests)]

    while engine.has_pending_requests():
        results = engine.step()
        for res in results.outputs:
            assert len(res.sequences) == 1
            seq = res.sequences[0]

            if seq.is_finished:
                assert seq.finish_reason == FinishReason.Length, seq.finish_reason
            else:
                generated[int(res.request_id)] += seq.delta


def _test_ignore_eos(
    engine,
    num_requests=5,
):
    prompt = "hi"
    s = 113
    requests = [
        create_request(
            idx=str(n - s),
            prompt=prompt,
            temp=0,
            freq_pen=0,
            pre_pen=0,
            max_tokens=n,
            stop=None,
            ignore_eos=True,
        )
        for n in range(s, s + num_requests)
    ]
    engine.add(requests)

    generated = ["" for _ in range(num_requests)]

    while engine.has_pending_requests():
        results = engine.step()
        for res in results.outputs:
            assert len(res.sequences) == 1
            seq = res.sequences[0]

            if seq.is_finished:
                assert (
                    seq.num_generated_tokens
                    == requests[int(res.request_id)].stopping_criteria.max_tokens
                )
                assert seq.finish_reason == FinishReason.Length
            else:
                generated[int(res.request_id)] += seq.delta


def _test_stop(
    engine,
    num_requests=5,
):
    prompt = "Write a merge sort program in Python."
    requests = []
    for n, stop in enumerate(["\n", ["\n"], "\n\n", "!", ["n", "!"]]):
        requests.append(
            create_request(
                idx=str(n),
                prompt=prompt,
                temp=0,
                freq_pen=0,
                pre_pen=0,
                max_tokens=300,
                stop=stop,
                ignore_eos=False,
            )
        )
    engine.add(requests)

    generated = ["" for _ in range(num_requests)]

    while engine.has_pending_requests():
        results = engine.step()
        for res in results.outputs:
            assert len(res.sequences) == 1
            seq = res.sequences[0]
            req_id = int(res.request_id)

            if seq.delta:
                generated[int(res.request_id)] += seq.delta

            if seq.is_finished:
                assert (
                    seq.finish_reason == FinishReason.Stop
                ), f"{seq.finish_reason.name}"
                gen_txt = generated[req_id]

                # stop token should appear only once in the gen text.
                found = sum(
                    [
                        gen_txt.count(str_stop)
                        for str_stop in requests[
                            req_id
                        ].stopping_criteria.stop_sequences
                    ]
                )
                assert found == 1, f"{gen_txt!r}, matches: {found}"


def _test_logprobs(
    engine,
    num_requests=10,
):
    prompt = "hi could you please implement merge sort?"

    requests = [
        create_request(
            idx=str(n),
            prompt=prompt,
            temp=0,
            freq_pen=0,
            pre_pen=0,
            max_tokens=300,
            stop=None,
            ignore_eos=True,
            top_logprobs=random.randint(1, 5),
            logprobs=True,
        )
        for n in range(num_requests)
    ]
    engine.add(requests)

    generated = ["" for _ in range(num_requests)]

    cnt = 0
    while engine.has_pending_requests():
        results = engine.step()
        cnt += 1
        print(cnt)
        for res in results.outputs:
            assert len(res.sequences) == 1
            seq = res.sequences[0]
            if cnt > 1:
                assert (
                    seq.finish_reason is not None
                    or len(seq.logprob_info[0].top_logprobs)
                    == requests[int(res.request_id)].sampling_params.top_logprobs
                )

            if seq.is_finished:
                assert (
                    seq.num_generated_tokens
                    == requests[int(res.request_id)].stopping_criteria.max_tokens
                )
                assert seq.finish_reason == FinishReason.Length
            else:
                generated[int(res.request_id)] += seq.delta


if __name__ == "__main__":
    parser = get_default_mlc_serve_argparser("test engine with samplers")
    args = parser.parse_args()
    postproc_mlc_serve_args(args)
    max_num_batched_tokens = 2048

    """
    # Test staging engines
    staging_engine = create_engine(
        args.model_artifact_path, max_num_batched_tokens, use_staging_engine=True
    )
    _test_max_tokens(staging_engine)
    _test_ignore_eos(staging_engine)
    _test_stop(staging_engine)
    print("logprob")
    _test_logprobs(staging_engine)
    print("Done")
    # These tests are broken since we are now imposing no length limit
    # if max_tokens = None. The tests do not finish in a reasonable time.
    # _test_max_context_length(staging_engine)
    staging_engine.stop()
    """

    # Test sync engines
    sync_engine = create_engine(
        args.model_artifact_path, max_num_batched_tokens, use_staging_engine=False
    )
    _test_max_tokens(sync_engine)
    _test_ignore_eos(sync_engine)
    _test_stop(sync_engine)
    _test_logprobs(sync_engine)
    # These tests are broken since we are now imposing no length limit
    # if max_tokens = None. The tests do not finish in a reasonable time.
    # _test_max_context_length(sync_engine)
import numpy as np
import os
from transformers import AutoTokenizer

import tvm
from tvm import relax

from mlc_llm.utils import split_transform_deploy_mod
from mlc_llm.transform import FuseTransposeMatmul

def smoothquant(args, mod, params, model_names):
    target = args.target
    target_kind = target.kind.default_keys[0]
    if target_kind != "cpu":
        for i in range(len(params)):
            params[i] = tvm.nd.array(params[i].numpy(), device=tvm.device(target_kind))

    dataset = _get_dummy_dataset(args.artifact_path, device=tvm.device(target.kind.default_keys[0]))
    with target, relax.quantize.sqconfig():
        print("[SmoothQuant] Run smoothing...")
        mod = relax.quantize.smooth(mod, params, model_names, dataset, extra_passes=FuseTransposeMatmul())
        print("[SmoothQuant] Run calibration and quantization...")
        mod = relax.quantize.quantize(mod, params, model_names, dataset, extra_passes=FuseTransposeMatmul())
    # Free memory:
    params.clear()
    return mod


def _get_dummy_dataset(artifact_path, device, num=3):
    prompts_dataset = [
        "The capital of Canada is",
        "2+2=?",
        "What is the capital of Russia?",
        "Who is the president of the USA?",
    ]

    dataset = []
    print("[SmoothQuant] Starting to initialize tokenizer...")
    tokenizer_path = os.path.join(artifact_path, "params")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print("[SmoothQuant] Initialization of tokenizer was completed...")
    for prompt in prompts_dataset:
        prompt_tokens = tokenizer.encode(prompt)
        data = tvm.nd.array(np.array([prompt_tokens]).astype("int32"), device=device)
        dataset.append(data)

    return dataset
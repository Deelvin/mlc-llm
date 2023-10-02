import os
import argparse
import numpy as np
from tqdm import tqdm
from enum import Enum
from transformers import AutoTokenizer
from typing import List, Dict, Any
from datasets import load_dataset

import tvm
from tvm import relax

import mlc_llm
from mlc_llm import utils
from mlc_llm.utils import load_params


# List of supported calibration datasets.
dataset_list = ["dummy", "piqa"]

def get_runtime_func(funcs: List[str], mod: tvm.IRModule):
    lowered_mod = relax.transform.LegalizeOps()(mod)
    target = tvm.target.Target.current(allow_none=False)
    target_kind = target.kind.default_keys[0]
    vm_device = tvm.device(target_kind)
    if target_kind != "cpu":
        with target:
            lowered_mod = tvm.tir.transform.DefaultGPUSchedule()(lowered_mod)
    exe = relax.build(lowered_mod, target)
    vm = relax.VirtualMachine(exe, vm_device)
    runtime_funcs = [vm[fname] for fname in funcs]
    return runtime_funcs[0] if len(funcs) == 1 else runtime_funcs


def to_device(params, device):
    return [param.copyto(device) for param in params]


def _accumulate_outlier_stat(stat, data):
    if stat is None:
        stat = [element.numpy() for element in data]
    else:
        assert len(data) == len(stat)
        for idx in range(len(stat)):
            stat[idx] = np.maximum(stat[idx], data[idx].numpy())
    return stat


def _accumulate_act_outlier_stat(stat: List[np.ndarray], data: List[tvm.nd.NDArray]):
    a_data = data[::2]
    return _accumulate_outlier_stat(stat, a_data)


def _accumulate_weight_outlier_stat(stat: List[np.ndarray], data: List[tvm.nd.NDArray]):
    # Optimization step: no need to accumulate weights for each new element in dataset since
    # weights are the same.
    if stat is not None:
        return stat
    w_data = data[1::2]
    return _accumulate_outlier_stat(stat, w_data)


def _calculate_scale_params(
    func_name: str,
    stats,
    config: Dict[str, Any],
    dev: tvm.runtime.Device,
):
    if stats[func_name] is None:
        return {}

    # scales = act_scales.pow(alpha) / weight_scales.pow(1-alpha)
    alpha = config["alpha"]
    a_stat, w_stat = stats[func_name]
    assert len(a_stat) == len(w_stat)
    idx = 0
    scale_params = {}
    for a_element, w_elemet in zip(a_stat, w_stat):
        assert a_element.shape == w_elemet.shape
        assert len(a_element.shape) == 1
        scales = np.power(a_element, alpha) / np.power(w_elemet, 1 - alpha)
        if scales.size - np.count_nonzero(scales) > 0:
            print("Warning: Smoothing: scales have zero value")
            scales = np.ones_like(scales)
            assert False, "Not supported case: please, add more elements in dataset. Otherwise, NaNs in output are possible."
        scale_params[f"sq_scale_{idx}"] = tvm.nd.array(scales, dev)
        scale_params[f"sq_scale_{idx+1}"] = tvm.nd.array(scales, dev)
        idx += 2

    return scale_params


# Quantization algorithm for activations or weights.
class QAlgo(Enum):
    PER_CHANNEL = 1,
    PER_TENSOR = 2,


def _calculate_quant_scale_params(
        func_name: str,
        stats,
        config: Dict[str, Any],
        dev: tvm.runtime.Device,
):
    if stats[func_name] is None:
        return {}

    a_dtype = config["adtype"]
    w_dtype = config["wdtype"]
    a_qscheme = QAlgo.PER_TENSOR
    w_qscheme = QAlgo.PER_CHANNEL if config["qscheme"] == "smq_q8i8f16_1" else QAlgo.PER_TENSOR

    def _calculate_quant_scale(arr: np.ndarray, dtype: str, algo: str) -> np.ndarray:
        if algo is QAlgo.PER_CHANNEL:
            scale = arr / arr.dtype.type(np.iinfo(dtype).max)
        else:
            assert algo is QAlgo.PER_TENSOR
            scale = np.array([np.max(arr) / arr.dtype.type(np.iinfo(dtype).max)])
        return scale

    idx = 0
    scale_params = {}
    for a_element, w_element in zip(*stats[func_name]):
        a_scale = _calculate_quant_scale(a_element, a_dtype, a_qscheme)
        scale_params[f"sq_scale_{idx}"] = tvm.nd.array(a_scale, dev)
        w_scale = _calculate_quant_scale(w_element, w_dtype, w_qscheme)
        scale_params[f"sq_scale_{idx+1}"] = tvm.nd.array(w_scale, dev)
        idx += 2

    return scale_params


def _smooth(
    mod: tvm.IRModule,
    params: List[tvm.nd.NDArray],
    funcs: List[str],
    dataset: List[tvm.nd.NDArray],
    config: Dict[str, Any],
):
    mod = mlc_llm.transform.SmoothQuantAnnotator()(mod)
    stat_mod = mlc_llm.transform.SmoothQuantStatCollector()(mod)
    stat_mod = mlc_llm.transform.FuseTransposeMatmul()(stat_mod)

    prefill, decode, kvc, _, _ = get_runtime_func(funcs, stat_mod)

    # Calculate max statistics
    a_stat: List[np.ndarray] = None
    w_stat: List[np.ndarray] = None

    target = tvm.target.Target.current(allow_none=False)
    for data in tqdm(dataset, desc="Smoothing"):
        # Create KV-cache
        kv_caches = kvc()
        num_tokens = data.shape[1]
        seq_len_shape = tvm.runtime.ShapeTuple([num_tokens])

        # Run Encoder and update statistics for activations/weights
        (logits, kv_caches), outputs = prefill(data, seq_len_shape, kv_caches, *params)
        a_stat = _accumulate_act_outlier_stat(a_stat, outputs)
        w_stat = _accumulate_weight_outlier_stat(w_stat, outputs)

        # Run Decoder and update statistics for activations/weights
        for _ in range(config["decoder_invoke_num"]):
            # TODO: support softmax with temperature.
            logits_max = np.argmax(logits.numpy(), axis=-1).astype("int32")
            if logits_max[0] in config["stop_tokens"]:
                break
            next_token = tvm.nd.array(logits_max, device=tvm.device(target.kind.default_keys[0]))
            num_tokens += logits_max.shape[1]
            seq_len_shape = tvm.runtime.ShapeTuple([num_tokens])
            (logits, kv_caches), outputs = decode(next_token, seq_len_shape, kv_caches, *params)
            a_stat = _accumulate_act_outlier_stat(a_stat, outputs)
            w_stat = _accumulate_weight_outlier_stat(w_stat, outputs)

    # Use the same statistics for "prefill"/"decode"
    stat = dict.fromkeys(funcs)
    stat["prefill"] = (a_stat, w_stat)
    stat["decode"] = (a_stat, w_stat)
    for fname in funcs:
        scale_params = _calculate_scale_params(fname, stat, config, tvm.cpu(0))
        mod = relax.transform.BindParams(fname, scale_params)(mod)
    return mod


def _calibrate(
    mod: tvm.IRModule,
    params: List[tvm.nd.NDArray],
    funcs: List[str],
    dataset: List[tvm.nd.NDArray],
    config: Dict[str, Any],
):
    qscheme: str = config["qscheme"]
    mod = mlc_llm.transform.SmoothQuantAnnotator(qscheme)(mod)
    stat_mod = mlc_llm.transform.SmoothQuantStatCollector(qscheme)(mod)
    stat_mod = mlc_llm.transform.FuseTransposeMatmul()(stat_mod)

    prefill, decode, kvc, _, _ = get_runtime_func(funcs, stat_mod)

    # Calculate max statistics
    a_stat: List[np.ndarray] = None
    w_stat: List[np.ndarray] = None

    target = tvm.target.Target.current(allow_none=False)
    for data in tqdm(dataset, desc="Calibration"):
        # Create KV-cache
        kv_caches = kvc()
        num_tokens = data.shape[1]
        seq_len_shape = tvm.runtime.ShapeTuple([num_tokens])

        # Run Encoder and update statistics for activations/weights
        (logits, kv_caches), outputs = prefill(data, seq_len_shape, kv_caches, *params)
        a_stat = _accumulate_act_outlier_stat(a_stat, outputs)
        w_stat = _accumulate_weight_outlier_stat(w_stat, outputs)

        # Run Decoder and update statistics for activations/weights
        for _ in range(config["decoder_invoke_num"]):
            # TODO: support softmax with temperature.
            logits_max = np.argmax(logits.numpy(), axis=-1).astype("int32")
            if logits_max[0] in config["stop_tokens"]:
                break
            next_token = tvm.nd.array(logits_max, device=tvm.device(target.kind.default_keys[0]))
            num_tokens += logits_max.shape[1]
            seq_len_shape = tvm.runtime.ShapeTuple([num_tokens])
            (logits, kv_caches), outputs = decode(next_token, seq_len_shape, kv_caches, *params)
            a_stat = _accumulate_act_outlier_stat(a_stat, outputs)
            w_stat = _accumulate_weight_outlier_stat(w_stat, outputs)

    # Use the same statistics for "prefill"/"decode"
    stat = dict.fromkeys(funcs)
    stat["prefill"] = (a_stat, w_stat)
    stat["decode"] = (a_stat, w_stat)
    for fname in funcs:
        scale_params = _calculate_quant_scale_params(fname, stat, config, tvm.cpu(0))
        mod = relax.transform.BindParams(fname, scale_params)(mod)

    mod = mlc_llm.transform.SmoothQuantOpConverter(qscheme)(mod)
    mod = mlc_llm.transform.SmoothQuantLegalizer(qscheme, config["adtype"], config["wdtype"])(mod)
    mod = relax.transform.DeadCodeElimination(funcs)(mod)
    return mod


def smoothquant(args, mod, model_names):
    target = args.target
    smq_device = tvm.device(target.kind.default_keys[0])
    assert args.build_model_only is False, "build_model_only in True is not supported in SMQ"
    params = load_params(args.artifact_path, device=smq_device)

    dataset, stop_tokens = _get_dataset(args.dataset, args.artifact_path, device=smq_device)
    smq_config: Dict[str, Any] = {}
    smq_config["decoder_invoke_num"] = 5
    smq_config["alpha"] = 0.5
    smq_config["stop_tokens"] = stop_tokens
    smq_config["adtype"] = "int8"
    smq_config["wdtype"] = "int8"
    smq_config["qscheme"] = args.quantization.name
    with target:
        print("[SmoothQuant] Run smoothing...")
        mod = _smooth(mod, params, model_names, dataset, smq_config)
        print("[SmoothQuant] Run calibration and quantization...")
        mod = _calibrate(mod, params, model_names, dataset, smq_config)
    # Free memory:
    params.clear()
    print("[SmoothQuant] Smoothing and calibration were done!")
    return mod


def smoothquant_transform_params(
    args: argparse.Namespace,
    mod_transform: tvm.IRModule,
    params: List[tvm.nd.NDArray] = None,
) -> List[tvm.nd.NDArray]:
    mod_transform = relax.transform.ToNonDataflow()(mod_transform)
    mod_transform = relax.transform.LazyTransformParams()(mod_transform)

    target = args.target
    smq_device=tvm.device(target.kind.default_keys[0])

    new_params: List[tvm.nd.NDArray] = []
    if params is None:
        assert args.build_model_only is False, "build_model_only in True is not supported in SMQ"
        params = load_params(args.artifact_path, device=smq_device)

    @tvm.register_func("get_item", override=True)
    def get_item(i):
        assert params[i] is not None
        return tvm.nd.array(params[i], device=smq_device)

    @tvm.register_func("set_item", override=True)
    def set_item(i, value):
        if len(new_params) <= i:
            new_params.extend([None for _ in range(i - len(new_params) + 1)])
        new_params[i] = tvm.nd.array(value, device=tvm.cpu())

    if target.kind.name != "llvm":
        with tvm.target.Target(target):
            mod_transform = tvm.tir.transform.DefaultGPUSchedule()(mod_transform)

    assert "decode_transform_params" in [gv.name_hint for gv in mod_transform.get_global_vars()]
    ex = relax.build(mod_transform, target=target)
    vm = relax.vm.VirtualMachine(ex, smq_device)
    vm["decode_transform_params"]()
    return new_params


def smoothquant_quantize_params(
    mod: tvm.IRModule,
    model_names: List[str],
    args: argparse.Namespace,
):
    mod = relax.transform.LiftTransformParams()(mod)
    mod_transform, mod_deploy = utils.split_transform_deploy_mod(mod, model_names)
    new_params = smoothquant_transform_params(args, mod_transform)
    return mod_deploy, new_params


def _get_dummy_dataset(artifact_path, device, num=3):
    prompts_dataset = [
        "The capital of Canada is",
        "2+2=?",
        "What is the capital of France?",
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
    stop_tokens = ([tokenizer.eos_token_id])

    return dataset, stop_tokens


def _prepare_dummy_dataset(artifact_path: str):
    """
    Return path to file with dummy dataset. This dataset consists of 4 simple questions.
    "train", "test" and "validation" datasets use the same dummy file.
    """
    file_name = "dummy_calibration_dataset.txt"
    file_path = os.path.join(artifact_path, file_name)
    data_files = {"train": file_path, "test": file_path, "validation": file_path}
    if os.path.exists(file_path):
        return data_files

    prompts = [
        "The capital of Canada is",
        "2+2=?",
        "What is the capital of France?",
        "Who is the president of the USA?",
    ]
    f = open(file_path, "w+")
    for prompt in prompts:
        f.write(prompt + "\n")
    return data_files


def _get_dataset(name: str, artifact_path: str, device: tvm.runtime.Device):
    print("[SmoothQuant] Starting to initialize tokenizer...")
    tokenizer_path = os.path.join(artifact_path, "params")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    print("[SmoothQuant] Initialization of tokenizer was completed...")

    if name not in dataset_list:
        raise ValueError(f"Dataset {name} is not supported")
    if name == "piqa":
        data_files = None
        text_name = "goal"
    else:
        # Dummy dataset consisting of 4 simple questions.
        name = text_name = "text"
        data_files = _prepare_dummy_dataset(artifact_path)

    dataset = load_dataset(name, data_files=data_files, split="validation")
    calibration_dataset = []
    for record in dataset:
        data = tokenizer(record[text_name], return_tensors="np")
        calibration_dataset.append(tvm.nd.array(data["input_ids"].astype("int32"), device=device))
    stop_tokens = ([tokenizer.eos_token_id])

    return calibration_dataset, stop_tokens

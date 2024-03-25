"""Functions for pre-sharding weights"""
from typing import Any, Dict, List, Union
from collections import OrderedDict
import numpy as np
import os

from tvm import IRModule
from tvm import dlight as dl
from tvm import relax
from tvm import nd
from tvm.relax.frontend import nn
from tvm.runtime import Device, NDArray
from tvm.target import Target
from mlc_llm.support import tensor_parallel as tp

logger = logging.getLogger("preshard")


def _sharded_param_name(param_name, worker_id):
    return f"{param_name}_shard-{worker_id}"


def _update_quantize_map(
    quantize_map: Any,
    named_params: Dict[str, nn.Parameter],
    mlc_name: str,
    tensor_parallel_shards: int,
):
    param_names: List[str] = [mlc_name]

    if mlc_name in quantize_map.param_map:
        # the parameter is quantized
        quantized_params = quantize_map.param_map[mlc_name]
        param_names = quantized_params
        quantize_func = quantize_map.map_func[mlc_name]

        for worker_id in range(tensor_parallel_shards):
            sharded_mlc_name = _sharded_param_name(mlc_name, worker_id)
            quantize_map.param_map[sharded_mlc_name] = [
                _sharded_param_name(param_name, worker_id) for param_name in quantized_params
            ]
            quantize_map.map_func[sharded_mlc_name] = quantize_func

    for param_name in param_names:
        param = named_params.pop(param_name)
        for worker_id in range(tensor_parallel_shards):
            named_params[_sharded_param_name(param_name, worker_id)] = param


def create_shard_func(
    bb: relax.BlockBuilder,
    param: nn.Parameter,
    tensor_parallel_shards: int,
    do_split: bool = True,
):  # pylint: disable=too-many-locals
    shard_strategy = param.attrs.get("shard_strategy", None)
    # generate tir shard function
    tir_func = shard_strategy.gen_tir(shards=tensor_parallel_shards, weight=param)
    tir_func = tir_func.with_attr("global_symbol", f"{shard_strategy.name}_tir")
    # add tir shard function to the IRModule
    tir_gvar = bb.add_func(tir_func, func_name=f"{shard_strategy.name}_tir")
    # create relax function that
    #     1. shard weight with tir shard function, result: [num_shards, *sharded_weight_shape]
    #     2. split the sharded weight along dim 0, result: num_shards * [1, *sharded_weight_shape]
    #     3. squeeze the 0th-dim of all shards, result: num_shards * [*sharded_weight_shape]
    weight_shape = param.shape
    weight_shape[shard_strategy.dim] = weight_shape[shard_strategy.dim] * tensor_parallel_shards
    sharded_weight_shape = [tensor_parallel_shards, *param.shape]
    weight_var = relax.Var("weight", relax.TensorStructInfo(weight_shape, param.dtype))

    with bb.function(name=shard_strategy.name, params=[weight_var]):
        with bb.dataflow():
            lv0 = bb.emit(
                relax.call_tir(
                    tir_gvar,
                    weight_var,
                    out_sinfo=relax.TensorStructInfo(sharded_weight_shape, param.dtype),
                )
            )
            if do_split:
                lv1 = bb.emit(
                    relax.op.split(lv0, indices_or_sections=tensor_parallel_shards, axis=0)
                )
                output_vars = []
                for i in range(tensor_parallel_shards):
                    lvi = bb.emit(relax.TupleGetItem(lv1, i))
                    squeezed_lvi = bb.emit(relax.op.squeeze(lvi, 0))
                    output_vars.append(squeezed_lvi)
                gv = bb.emit_output(output_vars)
            else:
                gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)

    return tir_gvar, weight_shape, sharded_weight_shape


def _compile_shard_funcs(mod: IRModule, device: Device):
    target = Target.from_device(device)
    with target:
        mod = relax.transform.LegalizeOps()(mod)
        mod = dl.ApplyDefaultSchedule(  # type: ignore   # pylint: disable=not-callable
            dl.gpu.Matmul(),
            dl.gpu.GEMV(),
            dl.gpu.Reduction(),
            dl.gpu.GeneralReduction(),
            dl.gpu.Fallback(),
        )(mod)
    ex = relax.build(mod, target=target)
    vm = relax.VirtualMachine(ex, device)
    return vm


def apply_preshard(
    quantize_map: Any,
    named_params: Dict[str, nn.Parameter],
    tensor_parallel_shards: int,
    args: Any,
) -> Tuple[Dict[str, nn.Parameter], Dict[str, Callable[[NDArray], Sequence[NDArray]]]]:
    """Apply pre-sharding to the named parameters.

    Parameters
    ----------
    named_params : Dict[str, nn.Parameter]
        The named parameters of the model. If the model is quantized, the named parameters should
        the state dictionary of the quantized model.
    tensor_parallel_shards : int
        The number of tensor parallel shards.
    args : Any
        The parsed arguments of weight conversion.

    Returns
    -------
    Tuple[Dict[str, nn.Parameter], Dict[str, Callable[[NDArray], Sequence[NDArray]]]
        The updated named parameters and the mapping from parameter name to the shard function.
    """

    # Update quantize_map and named_params, create shard functions based on shard strategies.
    model_config = args.model.config.from_file(args.config)
    model_config.tensor_parallel_shards = tensor_parallel_shards
    model = args.model.model(model_config)
    model.to(args.quantization.model_dtype)

    bb = relax.BlockBuilder()
    param_to_shard_func = {}
    shard_func_names = set()
    has_shard_strategy = False
    for name, param in model.state_dict().items():
        shard_strategy = param.attrs.get("shard_strategy", None)
        if shard_strategy is not None:
            has_shard_strategy = True
            _update_quantize_map(quantize_map, named_params, name, tensor_parallel_shards)

            # create shard functions
            param_to_shard_func[name] = shard_strategy.name
            if shard_strategy.name not in shard_func_names:
                if not isinstance(shard_strategy, tp.ShardScalar):
                    create_shard_func(bb, param, tensor_parallel_shards)
                    shard_func_names.add(shard_strategy.name)

    if not has_shard_strategy:
        logger.warning(
            "No parameters with 'shard_strategy' found."
            "At least one parameter must have a 'shard_strategy' for presharding. "
            "The model will continue to convert weights in a non-presharded manner."
        )

    mod = bb.finalize()
    vm = _compile_shard_funcs(mod, args.device)

    for name in param_to_shard_func:
        param_to_shard_func[name] = vm[param_to_shard_func[name]]
    return param_to_shard_func

def _split_array(arr, num: int):
    return np.split(arr.numpy(), num) if arr is not None else [None] * num

def _duplicate_array(arr, num: int):
    return [np.copy(arr.numpy()) for _ in range(num)] if arr is not None else [None] * num

def shard_smoothquant_params(tensor_parallel_shards, args):
    model_config = args.model.config.from_file(args.config)
    model_config.tensor_parallel_shards = tensor_parallel_shards
    model = args.model.model(model_config)
    model.to(args.quantization.model_dtype)

    pth = args.statistics_path
    param_to_smooth_factor = load_file(path=os.path.join(pth, "smooth_scale2param.json"))
    param_to_scale = load_file(path=os.path.join(pth, "quantize_scale2param.json"))
    import tvm
    from tvm.contrib import tvmjs
    smoothing_factors_dict, _ = tvmjs.load_ndarray_cache(f"{pth}/smooth/", tvm.cpu())
    scales_dict, _ = tvmjs.load_ndarray_cache(f"{pth}/quantize/", tvm.cpu())

    out = OrderedDict()
    for name, param in model.state_dict().items():
        smooth_factor_names = param_to_smooth_factor["prefill"].pop(name, None)
        scale_names = param_to_scale["prefill"].pop(name, None)
        shard_strategy = param.attrs.get("shard_strategy", None)
        if smooth_factor_names is not None and scale_names is not None:
            a_factor, w_factor = smooth_factor_names
            a_scale, w_scale, a_zp, w_zp = scale_names
            if shard_strategy is not None:
                if shard_strategy.dim == 0:
                    a_factors = _duplicate_array(smoothing_factors_dict[a_factor], tensor_parallel_shards)
                    w_factors = _duplicate_array(smoothing_factors_dict[w_factor], tensor_parallel_shards)
                    a_scales =  _duplicate_array(scales_dict[a_scale], tensor_parallel_shards)
                    w_scales =  _split_array(scales_dict[w_scale], tensor_parallel_shards)
                    a_zps =     _duplicate_array(scales_dict[a_zp], tensor_parallel_shards)
                    w_zps =     _split_array(scales_dict[w_zp], tensor_parallel_shards)
                else:
                    assert shard_strategy.dim == 1, f"Not supported shard.dim={shard_strategy.dim}"
                    a_factors = _split_array(smoothing_factors_dict[a_factor], tensor_parallel_shards)
                    w_factors = _split_array(smoothing_factors_dict[w_factor], tensor_parallel_shards)
                    a_scales =  _split_array(scales_dict[a_scale], tensor_parallel_shards)
                    w_scales =  _duplicate_array(scales_dict[w_scale], tensor_parallel_shards)
                    a_zps =     _split_array(scales_dict[a_zp], tensor_parallel_shards)
                    w_zps =     _duplicate_array(scales_dict[w_zp], tensor_parallel_shards)
                for shard_idx in range(tensor_parallel_shards):
                    out[_sharded_param_name(a_factor, shard_idx)] = a_factors[shard_idx]
                    out[_sharded_param_name(w_factor, shard_idx)] = w_factors[shard_idx]
                    if args.quantization.name != "smq_q8i8f16_0" and \
                        args.quantization.name != "smq_e4m3_float8_0":
                        out[_sharded_param_name(w_scale, shard_idx)] = w_scales[shard_idx]
                        out[_sharded_param_name(w_zp, shard_idx)] = w_zps[shard_idx]
            else:
                out[a_factor] = smoothing_factors_dict[a_factor]
                out[w_factor] = smoothing_factors_dict[w_factor]
                if args.quantization.name != "smq_q8i8f16_0" and \
                    args.quantization.name != "smq_e4m3_float8_0":
                    out[w_scale]  = scales_dict[w_scale]
                    out[w_zp]  = scales_dict[w_zp]
    return out

def load_file(path):
    import json
    with open(path, 'r') as f:
        loaded_dict = json.load(f)
    return loaded_dict

def _create_smoothquant_func(
    bb: relax.BlockBuilder, param: nn.Parameter, param_name: str, idx: int, tensor_parallel_shards: int, **smq_params
):
    def _create_func(
        func_name: str,
        bb: relax.BlockBuilder,
        param: nn.Parameter,
        smoothing_factor: Union[np.ndarray, nd.NDArray],
        scale: Union[np.ndarray, nd.NDArray],
        zp: Union[np.ndarray, nd.NDArray],
        dtype: str
    ):
        weight_var = relax.Var("weight", relax.TensorStructInfo(param.shape, param.dtype))
        with bb.function(name=func_name, params=[weight_var]):
            with bb.dataflow():
                if smoothing_factor is not None:
                    weight_var = bb.emit(relax.op.multiply(weight_var, relax.const(smoothing_factor)))
                if scale is not None and zp is not None:
                    weight_var = bb.emit(relax.op.quantize(weight_var, relax.const(scale), relax.const(zp), axis=-2, out_dtype=dtype))
                gv = bb.emit_output(weight_var)
            bb.emit_func_output(gv)

    func_names = []
    shard_strategy = param.attrs.get("shard_strategy", None)
    factor_param = smq_params.get("smoothing_factor")
    scale_param = smq_params.get("scale")
    zp_param = smq_params.get("zp")
    if tensor_parallel_shards == 1 or shard_strategy is None:
        func_name = f"convert_param_{idx}"
        func_names.append((param_name, func_name))
        _create_func(func_name, bb, param, factor_param, scale_param, zp_param, smq_params.get("quant_config").quantize_dtype)
    else:
        if shard_strategy.dim == 0:
            factors = _duplicate_array(factor_param, tensor_parallel_shards)
            scales =  _split_array(scale_param, tensor_parallel_shards)
            zps =     _split_array(zp_param, tensor_parallel_shards)
        else:
            assert shard_strategy.dim == 1, f"Not supported shard.dim={shard_strategy.dim}"
            factors = _split_array(factor_param, tensor_parallel_shards)
            scales =  _duplicate_array(scale_param, tensor_parallel_shards)
            zps =     _duplicate_array(zp_param, tensor_parallel_shards)
        for shard_idx in range(tensor_parallel_shards):
            func_name = f"convert_param_{idx}_shard_{shard_idx}"
            func_names.append((_sharded_param_name(param_name, shard_idx), func_name))
            _create_func(func_name, bb, param, factors[shard_idx], scales[shard_idx], zps[shard_idx], dtype = smq_params.get("quant_config").quantize_dtype)
    return func_names

def gen_smoothquant(named_params: Dict[str, nn.Parameter], tensor_parallel_shards: int, args: Any):
    model_config = args.model.config.from_file(args.config)
    model_config.tensor_parallel_shards = tensor_parallel_shards
    model = args.model.model(model_config)
    model.to(args.quantization.model_dtype)
    pth = args.output
    param_to_smooth_factor = load_file(path=f"{args.statistics_path}/smooth_scale2param.json")
    param_to_scale = load_file(path=f"{args.statistics_path}/quantize_scale2param.json")
    import tvm
    from tvm.contrib import tvmjs
    smoothing_factors_dict, _ = tvmjs.load_ndarray_cache(f"{args.statistics_path}/smooth/", tvm.cpu())
    scales_dict, _ = tvmjs.load_ndarray_cache(f"{args.statistics_path}/quantize/", tvm.cpu())

    bb = relax.BlockBuilder()
    param_to_smoothquant_func = {}
    for idx, (name, param) in enumerate(model.state_dict().items()):
        smooth_factor_names = param_to_smooth_factor["prefill"].pop(name, None)
        scale_names = param_to_scale["prefill"].pop(name, None)
        if smooth_factor_names is not None and scale_names is not None:
            _, smooth_factor_name = smooth_factor_names
            _, scale_name, _, zp_name = scale_names
            func_names = _create_smoothquant_func(
                bb,
                param,
                name,
                idx,
                tensor_parallel_shards,
                smoothing_factor=smoothing_factors_dict[smooth_factor_name],
                scale=scales_dict[scale_name],
                zp=scales_dict[zp_name],
                quant_config = args.quantization,
                dtype = args.quantization.model_dtype
            )
            for sharded_param_name, func_name in func_names:
                param_to_smoothquant_func[sharded_param_name] = func_name

                named_params[sharded_param_name].to(args.quantization.quantize_dtype)  # Update dtype for checker

    assert not param_to_smooth_factor["prefill"], "detected not processed smoothing factors"
    assert not param_to_scale["prefill"], "detected not processed scales/zero_points"

    mod = bb.finalize()
    vm = _compile_shard_funcs(mod, args.device)

    for name in param_to_smoothquant_func:
        param_to_smoothquant_func[name] = vm[param_to_smoothquant_func[name]]
    return param_to_smoothquant_func

"""A centralized registry of all existing quantization methods and their configurations."""
from typing import Any, Dict

from .awq_quantization import AWQQuantize
from .ft_quantization import FTQuantize
from .group_quantization import GroupQuantize
from .no_quantization import NoQuantize
from .per_tensor_quantization import PerTensorQuantize

Quantization = Any
"""Quantization is an object that represents an quantization algorithm. It is required to
have the following fields:

    name : str
        The name of the quantization algorithm, for example, "q4f16_1".

    kind : str
        The kind of quantization algorithm, for example, "group-quant", "faster-transformer".

It is also required to have the following method:

    def quantize_model(self, module: nn.Module) -> nn.Module:
        ...

    def quantize_weight(self, weight: tvm.runtime.NDArray) -> List[tvm.runtime.NDArray]:
        ...
"""

QUANTIZATION: Dict[str, Quantization] = {
    "q0f16": NoQuantize(
        name="q0f16",
        kind="no-quant",
        model_dtype="float16",
    ),
    "q0f32": NoQuantize(
        name="q0f32",
        kind="no-quant",
        model_dtype="float32",
    ),
    "q3f16_0": GroupQuantize(
        name="q3f16_0",
        kind="group-quant",
        group_size=40,
        quantize_dtype="int3",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="KN",
        quantize_embedding=True,
        quantize_final_fc=True,
    ),
    "q3f16_1": GroupQuantize(
        name="q3f16_1",
        kind="group-quant",
        group_size=40,
        quantize_dtype="int3",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="NK",
        quantize_embedding=True,
        quantize_final_fc=True,
    ),
    "q4f16_0": GroupQuantize(
        name="q4f16_0",
        kind="group-quant",
        group_size=32,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="KN",
        quantize_embedding=True,
        quantize_final_fc=True,
    ),
    "q4f16_1": GroupQuantize(
        name="q4f16_1",
        kind="group-quant",
        group_size=32,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="NK",
        quantize_embedding=True,
        quantize_final_fc=True,
    ),
    "q4f32_1": GroupQuantize(
        name="q4f32_1",
        kind="group-quant",
        group_size=32,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float32",
        linear_weight_layout="NK",
        quantize_embedding=True,
        quantize_final_fc=True,
    ),
    "q4f16_2": GroupQuantize(
        name="q4f16_2",
        kind="group-quant",
        group_size=32,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="NK",
        quantize_embedding=False,
        quantize_final_fc=False,
    ),
    "q4f16_autoawq": AWQQuantize(
        name="q4f16_autoawq",
        kind="awq",
        group_size=128,
        quantize_dtype="int4",
        storage_dtype="uint32",
        model_dtype="float16",
    ),
    "q4f16_ft": FTQuantize(
        name="q4f16_ft",
        kind="ft-quant",
        quantize_dtype="int4",
        storage_dtype="int8",
        model_dtype="float16",
    ),
    "qf8f16_e4m3_0": GroupQuantize(
        name="qf8f16_e4m3_0",
        kind="group-quant",
        group_size=64,
        quantize_dtype="e4m3_float8",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="NK",
    ),
    "qf8f16_e5m2_0": GroupQuantize(
        name="qf8f16_e5m2_0",
        kind="group-quant",
        group_size=64,
        quantize_dtype="e5m2_float8",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="NK",
    ),
    "fp8_e5m2_e5m2": GroupQuantize(
        name="fp8_e5m2_e5m2",
        kind="group-quant",  # TODO(csullivan): Can I change this and still use GroupQuantize?
        group_size=64,
        quantize_dtype="e5m2_float8",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="NK",
    ),
    "fp8_e4m3_e4m3": GroupQuantize(
        name="fp8_e4m3_e4m3",
        kind="group-quant",  # TODO(csullivan): Can I change this and still use GroupQuantize?
        group_size=64,
        quantize_dtype="e4m3_float8",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="NK",
    ),
    "fp8_e4m3_e5m2": GroupQuantize(
        name="fp8_e4m3_e5m2",
        kind="group-quant",  # TODO(csullivan): Can I change this and still use GroupQuantize?
        group_size=64,
        quantize_dtype="e5m2_float8",
        storage_dtype="uint32",
        model_dtype="float16",
        linear_weight_layout="NK",
    ),
    "fp8_e4m3_e5m2_max_calibration": PerTensorQuantize(
        name="fp8_e4m3_e5m2_max_calibration",
        kind="per-tensor-quant",
        activation_dtype="e4m3_float8",
        weight_dtype="e5m2_float8",
        storage_dtype="uint32",
        model_dtype="float16",
        no_scale=True,
        quantize_embedding=False,
    ),
    "fp8_e4m3_e5m2_max": PerTensorQuantize(
        name="fp8_e4m3_e5m2_max",
        kind="per-tensor-quant",
        activation_dtype="e4m3_float8",
        weight_dtype="e5m2_float8",
        storage_dtype="uint32",
        model_dtype="float16",
        no_scale=True,
        quantize_embedding=False,
    ),
    "fp8_e4m3_e4m3_max_calibration": PerTensorQuantize(
        name="fp8_e4m3_e4m3_max_calibration",
        kind="per-tensor-quant",
        activation_dtype="e4m3_float8",
        weight_dtype="e4m3_float8",
        storage_dtype="uint32",
        model_dtype="float16",
        quantize_embedding=False,
    ),
    "fp8_e4m3_e4m3_max": PerTensorQuantize(
        name="fp8_e4m3_e4m3_max",
        kind="per-tensor-quant",
        activation_dtype="e4m3_float8",
        weight_dtype="e4m3_float8",
        storage_dtype="uint32",
        model_dtype="float16",
        quantize_embedding=False,
    ),
    "fp16_max_calibration": PerTensorQuantize(
        name="fp16_max_calibration",
        kind="per-tensor-quant",
        activation_dtype="float16",
        weight_dtype="float16",
        storage_dtype="float16",
        model_dtype="float16",
        quantize_embedding=False,
        no_scale=True,
    ),
}

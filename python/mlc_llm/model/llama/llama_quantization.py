"""This file specifies how MLC's Llama parameters are quantized using group quantization
or other formats."""
from typing import Tuple

from tvm.relax.frontend import nn

from mlc_llm.loader import QuantizeMapping
from mlc_llm.quantization import (
    AWQQuantize,
    FTQuantize,
    GroupQuantize,
    NoQuantize,
    SmoothQuantize,
    PreprocessSmoothQuantize,
)

from .llama_model import LlamaConfig, LlamaForCasualLM


def group_quant(
    model_config: LlamaConfig,
    quantization: GroupQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llama-architecture model using group quantization."""
    model: nn.Module = LlamaForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def ft_quant(
    model_config: LlamaConfig,
    quantization: FTQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llama-architecture model using FasterTransformer quantization."""
    model: nn.Module = LlamaForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def awq_quant(
    model_config: LlamaConfig,
    quantization: AWQQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llama-architecture model using Activation-aware Weight Quantization(AWQ)."""
    model: nn.Module = LlamaForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def no_quant(
    model_config: LlamaConfig,
    quantization: NoQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llama2 model without quantization."""
    model: nn.Module = LlamaForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    return model, quant_map


def smooth_quant(
    model_config: LlamaConfig,
    quantization: SmoothQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Quantize a Llama-architecture model using SmoothQuant."""
    model: nn.Module = LlamaForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map


def prequantize_smooth(
    model_config: LlamaConfig,
    quantization: PreprocessSmoothQuantize,
) -> Tuple[nn.Module, QuantizeMapping]:
    """Preprocessing for smooth quantization for  Llama-architecture model."""
    model: nn.Module = LlamaForCasualLM(model_config)
    model.to(quantization.model_dtype)
    quant_map = QuantizeMapping({}, {})
    model = quantization.quantize_model(
        model,
        quant_map,
        "",
    )
    return model, quant_map

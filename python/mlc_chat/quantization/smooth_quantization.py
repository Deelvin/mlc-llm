"""The SmoothQuant config"""

from dataclasses import dataclass
from typing import List, Literal

from tvm.relax.frontend import nn
from tvm.runtime import NDArray

from ..loader import QuantizeMapping

@dataclass
class SmoothQuantize:  # pylint: disable=too-many-instance-attributes
    name: str
    kind: str
    activation_dtype: Literal["int8", "e4m3_float8", "e5m2_float8"]
    weight_dtype: Literal["int8", "e4m3_float8", "e5m2_float8"]
    zero_point_dtype: Literal["int8", "float16", "float16"]
    accumulator_dtype: Literal["int32", "float32", "float32"]
    model_dtype: Literal["float16"]

    def __post_init__(self):
        return

    def quantize_weight(self, weight: NDArray) -> List[NDArray]:
        return []

    def quantize_model(
        self,
        model: nn.Module,
        quant_map: QuantizeMapping,
        name_prefix: str,
    ) -> nn.Module:
        """
        Quantize model with using smooth quantization
        currently all conversions are performed using compilation passes.
        ToDo: apply final model conversion using this pass.

        Parameters
        ----------
        model : nn.Module
            The non-quantized nn.Module.

        quant_map : QuantizeMapping
            The quantize mapping with name mapping and func mapping.

        name_prefix : str
            The name prefix for visited weight.

        Returns
        -------
        ret : nn.Module
            The quantized nn.Module.
        """
        return model

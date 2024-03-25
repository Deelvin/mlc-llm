"""The SmoothQuant config"""

from dataclasses import dataclass
from typing import Any, Callable, List, Literal, Optional, Tuple

import tvm
from tvm import DataType, DataTypeCode, IRModule
from tvm import dlight as dl
from tvm import relax, te, tir
from tvm.relax.frontend import nn
from tvm.runtime import NDArray
from tvm.target import Target

from ..loader import QuantizeMapping
from ..op import faster_transformer_dequantize_gemm
from ..support import logging
from ..support.auto_target import detect_cuda_arch_list
from .group_quantization import GroupQuantize, GroupQuantizeEmbedding
# from .smoothquant_utils import smoothquant

from tvm.relax.frontend.nn import op

logger = logging.getLogger(__name__)

QSCHEMES = ("smq_q8i8f16_0", "smq_q8i8f16_1", "smq_q8i8f16_2", "smq_q8i8f32_2")
OPMODES = ("smoothing", *QSCHEMES)

XX = 0

@dataclass
class SmoothQuantize:  # pylint: disable=too-many-instance-attributes
  name: str
  kind: str
  quantize_dtype: Literal["int8"]
  storage_dtype: Literal["int8"]
  model_dtype: Literal["float16"]

  def __post_init__(self):
    return

  def quantize_weight(self, weight: NDArray) -> List[NDArray]:
     print("quantize_weight:")
     print(weight)
     return []

  def quantize_model(
        self,
        model: nn.Module,
        quant_map: QuantizeMapping,
        name_prefix: str,
    ) -> nn.Module:
      """
      Quantize model with FasterTransformer quantization

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

class SMQQuantizeLinear(nn.Module):  # pylint: disable=too-many-instance-attributes
    """An nn.Linear module with FasterTransformer quantization"""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_features: int,
        out_features: int,
        config: SmoothQuantize,
        bias: bool = True,
        out_dtype: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.config = config
        # cur_group_size = in_features if not config.group_size else config.group_size
        self.q_weight = nn.Parameter(
            (in_features, out_features),
            config.storage_dtype,
        )
        # self.q_scale = nn.Parameter(
        #     (tir.ceildiv(in_features, cur_group_size), out_features), config.model_dtype
        # )
        # if bias:
        #     self.bias = nn.Parameter(
        #         (out_features,), config.model_dtype if out_dtype is None else out_dtype
        #     )
        # else:
        #     self.bias = None

    @staticmethod
    def from_linear(src: nn.Linear, config: SmoothQuantize) -> "SmoothQuantizeLinear":
        """
        Converts a non-quantized nn.Linear to a FasterTransformer quantized FTQuantizeLinear

        Parameters
        ----------
        src : nn.Linear
            The non-quantized nn.Linear.

        config : FTQuantize
            The FasterTransformer quantization config.

        Returns
        -------
        ret : FTQuantizeLinear
            The FasterTransformer quantized FTQuantizeLinear layer.
        """
        bias=getattr(src, "bias", None) is not None
        quantized_linear = SMQQuantizeLinear(
            in_features=src.in_features,
            out_features=src.out_features,
            config=config,
            bias=bias,
            out_dtype=src.out_dtype,
        )
        print("src", src)
        print("src.in_features", src.in_features)
        print("src.in_features", src.out_features)
        print("bias", bias)
        # if quantized_linear.bias is not None:
        #     quantized_linear.bias.attrs = src.bias.attrs
        return quantized_linear

    def forward(self, x: nn.Tensor) -> nn.Tensor:  # pylint: disable=invalid-name
        """
        Forward method for FasterTransformer quantized linear layer.

        Parameters
        ----------
        x : nn.Tensor
            The input tensor.

        Returns
        -------
        ret : nn.Tensor
            The output tensor for the FasterTransformer quantized linear layer.
        """
        print(x.shape)
        
        return op.extern(
            name="tvm.contrib.cublas.hgem_matmul",
            args=[x, self.q_weight],
            out=nn.Tensor.placeholder((*x.shape[:-1], self.out_features), dtype="float16"),
          )

        # return faster_transformer_dequantize_gemm(
        #     x, self.q_weight, self.q_scale, self.bias, group_size=self.config.group_size
        # )

    def to(self, dtype: Optional[str] = None) -> None:
        """
        Override to() such that we do not convert bias if there is an out_dtype.
        Otherwise, we might run into dtype mismatch when computing x + self.bias.
        """
        print("TO call")
        self.q_weight.to(dtype=dtype)
        self.q_scale.to(dtype=dtype)
        if self.bias is not None and self.out_dtype is None:
            self.bias.to(dtype=dtype)
        if dtype is not None and isinstance(getattr(self, "dtype", None), str):
            self.dtype = dtype  # pylint: disable=attribute-defined-outside-init

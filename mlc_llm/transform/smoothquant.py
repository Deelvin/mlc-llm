import numpy as np
import tvm
from tvm import relax
from tvm.relax.analysis import remove_all_unused
from tvm.relax.dpl import rewrite_call, is_op, wildcard, is_tuple_get_item
from tvm.relax.expr_functor import PyExprMutator, mutator, visitor, PyExprVisitor
from tvm.script import relax as R

#from ..quantization.smoothquant_utils import (
#    SCALE_PREFIX_NAME, ZP_PREFIX_NAME, _try_convert_to_scalar_const
#)

from typing import Dict, List, Union
from enum import Enum


QSCHEMES = ("smq_q8i8f16_0", "smq_q8i8f16_1", "smq_q8i8f16_2", "smq_q8i8f32_2")
OPMODES = ("smoothing", *QSCHEMES)

SCALE_PREFIX_NAME = "sq_scale_"
ZP_PREFIX_NAME = "sq_zp_"
SMOOTH_SUFFIX_NAME = "smooth"
CALIBRATE_SUFFIX_NAME = "calibrate"
ZP_PREFIX_NAME = "sq_zp_"


def _try_convert_to_scalar_const(expr: tvm.relax.Expr) -> Union[tvm.relax.Expr, float, int]:
    if isinstance(expr, tvm.relax.Constant):
        if expr.struct_info.ndim == 0:
            return expr.data.numpy()[()].item()
        elif expr.struct_info.ndim == 1:
            dim_size = expr.struct_info.shape[0].value
            if dim_size == 1:
                return expr.data.numpy()[0].item()
    return expr


class QKind(Enum):
    KIND_ACT = 1,
    KIND_WEIGHTS = 2,

def get_scale_param_name(counter, suffix):
    return f"{SCALE_PREFIX_NAME}{counter}_{suffix}"

def get_zp_param_name(counter):
    return f"{ZP_PREFIX_NAME}{counter}"

@mutator
class Annotator(PyExprMutator):
    def __init__(self, irmod: tvm.IRModule, idx_to_param_name: Dict[int, str], op_mode: str) -> None:
        super().__init__(irmod)
        self.mod = irmod
        self.sm_counter = 0
        self.new_params = []
        # Operation mode of the annotator: "smoothing" or "quantization"
        assert op_mode in OPMODES, f"unsupported operation mode: '{op_mode}'"
        self.op_mode = op_mode
        self.idx_to_param_name = idx_to_param_name
        self.scale2param: Dict[str, Dict[str, str]] = {}
        self.curr_func_name = None

    def transform(self) -> tvm.IRModule:
        for gv, func in self.mod.functions.items():
            if not isinstance(func, relax.Function):
                continue
            self.curr_func_name = gv.name_hint
            self.scale2param[self.curr_func_name] = {}
            self.sm_counter = 0
            self.new_params = []
            updated_func = self.visit_expr(func)
            updated_func = remove_all_unused(updated_func)
            self.builder_.update_func(gv, updated_func)

        return self.builder_.get()

    def visit_function_(self, f):
        body = super().visit_expr(f.body)
        params = list(f.params) + list(self.new_params)
        return tvm.relax.Function(params, body, f.ret_struct_info, f.is_pure, f.attrs, f.span)

    def visit_call_(self, call: relax.Call) -> relax.Expr:
        call = super().visit_call_(call)
        if call.op != tvm.ir.Op.get("relax.matmul"):
            return call
        permute = self.lookup_binding(call.args[1])
        if permute is None or permute.op != tvm.ir.Op.get("relax.permute_dims"):
            return call
        act = call.args[0]
        weights = permute.args[0]
        if weights.struct_info.ndim != 2:
            return call
        # Skip linear ops with weights of dynamic shape
        if any([isinstance(dim, tvm.tir.Var) for dim in weights.struct_info.shape]):
            return call
        if act.struct_info.ndim != 2 and act.struct_info.ndim != 3:
            return call

        def _make_param(param_name: str, shape, dtype: str) -> tvm.relax.Var:
            param = relax.Var(param_name, relax.TensorStructInfo(shape, dtype))
            self.sm_counter += 1
            self.new_params.append(param)
            return param

        def _make_scale_param(shape: relax.ShapeExpr, dtype: str, kind: QKind, suffix: str) -> tvm.relax.Var:
            """
            Create scale parameter.

            In case of smoothing:
              scale is a 1D Tensor with the size of reduction axis for the given R.linear op
              (size == shape[-1] of activations/weights).

            In case of quantization:
              scale is a 1D Tensor with the size == shape[-1] for activations and size == shape[-2]
              for the weights. In case of per-tensor quantization scheme, single element is
              broadcasted to the corresponding size.
            """
            axis = -1
            if kind == QKind.KIND_WEIGHTS and self.op_mode.startswith("smq_q8i8"):
                axis = -2
            n = shape[axis]
            scale = _make_param(get_scale_param_name(self.sm_counter, suffix), shape=[n], dtype=dtype)
            return scale, axis

        def _make_zero_point_param(shape: relax.ShapeExpr, dtype: str) -> tvm.relax.Var:
            return _make_param(get_zp_param_name(self.sm_counter), shape=shape, dtype=dtype)

        if self.op_mode.startswith("smq_q8i8"):
            a_scale, a_axis = _make_scale_param(act.struct_info.shape, act.struct_info.dtype, kind=QKind.KIND_ACT, suffix=CALIBRATE_SUFFIX_NAME)
            w_scale, w_axis = _make_scale_param(weights.struct_info.shape, weights.struct_info.dtype, kind=QKind.KIND_WEIGHTS, suffix=CALIBRATE_SUFFIX_NAME)

            a_zp = _make_zero_point_param(a_scale.struct_info.shape, dtype="int8")
            w_zp = _make_zero_point_param(w_scale.struct_info.shape, dtype="int8")
            qa = R.quantize(act, a_scale, a_zp, axis=a_axis, out_dtype="int8")
            lhs = R.dequantize(qa, a_scale, a_zp, axis=a_axis, out_dtype=act.struct_info.dtype)
            qw = R.quantize(weights, w_scale, w_zp, axis=w_axis, out_dtype="int8")
            rhs = R.dequantize(qw, w_scale, w_zp, axis=w_axis, out_dtype=weights.struct_info.dtype)
            multiply = self.lookup_binding(weights)
            if self.idx_to_param_name is not None:
                tgi = self.lookup_binding(multiply.args[0])
                assert isinstance(tgi, relax.TupleGetItem), "SmoothQuantAnnotator: unsupported case"
                param_name = self.idx_to_param_name[tgi.index]
                self.scale2param[self.curr_func_name][param_name] = (
                    a_scale.name_hint, w_scale.name_hint, a_zp.name_hint, w_zp.name_hint
                )

        else:
            a_scale, a_axis = _make_scale_param(act.struct_info.shape, act.struct_info.dtype, kind=QKind.KIND_ACT, suffix=SMOOTH_SUFFIX_NAME)
            w_scale, w_axis = _make_scale_param(weights.struct_info.shape, weights.struct_info.dtype, kind=QKind.KIND_WEIGHTS, suffix=SMOOTH_SUFFIX_NAME)
            lhs = R.divide(act, a_scale)
            rhs = R.multiply(weights, w_scale)
            if self.idx_to_param_name is not None:
                tgi = self.lookup_binding(weights)
                assert isinstance(tgi, relax.TupleGetItem), "SmoothQuantAnnotator: unsupported case"
                param_name = self.idx_to_param_name[tgi.index]
                self.scale2param[self.curr_func_name][param_name] = (
                    a_scale.name_hint, w_scale.name_hint
                )

        return R.linear(lhs, rhs)


@tvm.transform.module_pass(opt_level=0, name="SmoothQuantAnnotator")
class SmoothQuantAnnotator:
    """
    Insert R.multiply + R.divide (or R.quantize + R.dequantize in case of op_mode == "smq_q8i8f16*")
    ops before R.linear. Add scales (second argument of R.multiply or R.smooth) or scales and
    zero_points (in case of quantization) to the list of relax.Function parameters.
    Example:
      R.linear(lhs, rhs)  -->  op1 = R.quantize(lhs, scale1, zero_point1)
                               op2 = R.dequantize(op1, scale1, zero_point1)
                               op3 = R.quantize(rhs, scale2, zero_point2)
                               op4 = R.dequantize(op3, scale2, zero_point2)
                               R.linear(op2, op4)

      for the self.op_mode == "smoothing" case:
      R.linear(lhs, rhs)  -->  op1 = R.divide(lhs, scale1)
                               op2 = R.multiply(rhs, scale2)
                               R.linear(op1, op2)
    """
    def __init__(self, index_to_pname: Dict[int, str] = None, op_mode: str = "smoothing") -> None:
        self.op_mode = op_mode
        self.index_to_pname = index_to_pname
        self.scale2param: Dict[str, Dict[str, str]] = {}

    def transform_module(self, irmod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.IRModule:
        module_ann = Annotator(irmod, self.index_to_pname, self.op_mode)
        new_irmod = module_ann.transform()
        self.scale2param.update(module_ann.scale2param)
        return new_irmod


@tvm.transform.module_pass(opt_level=0, name="SmoothQuantStatCollector")
class SmoothQuantStatCollector:
    """
    This pass modifies IRModule to enable statistics collection. It does several modifications:
    1) Insert chain of simple ops (abs, max, squeeze) just before R.linear op and remove annotate
       operation (R.quantize/R.dequantize/R.divide/R.multiply). This is done for memory footprint
       optimization only. Since we do not want to dump the whole tensor and dump already
       preprocessed information (abs -> max -> squeeze for smoothing, min/max for quantization).
    2) Remove scale and zero_point params from relax.Function.
    3) Add new outputs in relax.Function that correspond to the last op in the sequance from 1).
    """
    def transform_module(self, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.IRModule:
        @mutator
        class ParamsAndOutputsMutator(PyExprMutator):
            def __init__(self, mod: tvm.IRModule) -> None:
                super().__init__(mod)
                self.mod = mod
                self.var2val: Dict[relax.Var, relax.Expr] = {}
                self.profile_points = []
                self.params_to_remove = []

                self.input = wildcard()
                self.weights = wildcard()
                lhs_quantize = is_op("relax.quantize")(self.input, wildcard(), wildcard())
                self.lhs_sm = (
                    is_op("relax.dequantize")(lhs_quantize, wildcard(), wildcard()) |
                    is_op("relax.divide")(self.input, wildcard())
                )
                rhs_quantize = is_op("relax.quantize")(self.weights, wildcard(), wildcard())
                self.rhs_sm = (
                    is_op("relax.dequantize")(rhs_quantize, wildcard(), wildcard()) |
                    is_op("relax.multiply")(self.weights, wildcard())
                )
                self.permute = is_op("relax.permute_dims")(self.rhs_sm)
                self.pattern = is_op("relax.matmul")(self.lhs_sm, self.permute)

            def transform(self) -> tvm.IRModule:
                for gv, func in self.mod.functions.items():
                    if not isinstance(func, relax.Function):
                        continue
                    self.var2val = tvm.relax.analysis.get_var2val(func)
                    self.profile_points = []
                    self.params_to_remove = []
                    updated_func = self.visit_expr(func)
                    updated_func = remove_all_unused(updated_func)
                    self.builder_.update_func(gv, updated_func)
                return self.builder_.get()

            def visit_function_(self, f):
                body = super().visit_expr(f.body)
                new_params = [param for param in f.params if param not in self.params_to_remove]
                return relax.Function(new_params, body, None, f.is_pure, f.attrs, f.span)
            
            def visit_seq_expr_(self, op: relax.SeqExpr) -> relax.Expr:
                op = super().visit_seq_expr_(op)
                if len(self.profile_points) != 0:
                    new_body = relax.Tuple([op.body, *self.profile_points])
                    return relax.SeqExpr(op.blocks, new_body, op.span)
                return op

            def visit_dataflow_block_(self, block: relax.DataflowBlock) -> relax.DataflowBlock:
                self.builder_._begin_dataflow_block()
                for binding in block.bindings:
                    self.visit_binding(binding)
                # Mark all profile points as new outputs in the block.
                if len(self.profile_points) != 0:
                    self.profile_points = [self.builder_.emit_output(self.profile_points)]
                return self.builder_._end_block()

            def visit_call_(self, call: relax.Call) -> relax.Expr:
                call = super().visit_call_(call)
                matchings = self.pattern.extract_matched_expr(call, self.var2val)
                if matchings:
                    data = matchings[self.input]
                    weights = matchings[self.weights]
                    lhs_op = matchings[self.lhs_sm]
                    rhs_op = matchings[self.rhs_sm]
                    if lhs_op.op == tvm.ir.Op.get("relax.divide") and rhs_op.op == tvm.ir.Op.get("relax.multiply"):
                        a_out = self._emit_abs_max_ops_chain(data)
                        w_out = self._emit_abs_max_ops_chain(weights)
                        self.profile_points.extend([a_out, w_out])
                        self.params_to_remove.extend([lhs_op.args[1], rhs_op.args[1]])
                    else:
                        assert lhs_op.op == tvm.ir.Op.get("relax.dequantize")
                        assert rhs_op.op == tvm.ir.Op.get("relax.dequantize")
                        a_max_out, a_min_out = self._emit_max_min_ops_chain(data, axis=-2)
                        w_max_out, w_min_out = self._emit_max_min_ops_chain(weights, axis=-1)
                        self.profile_points.extend([a_max_out, a_min_out, w_max_out, w_min_out])
                        self.params_to_remove.extend(
                            [lhs_op.args[1], rhs_op.args[1], lhs_op.args[2], rhs_op.args[2]]
                        )
                    return self.builder_.emit(R.linear(data, weights))

                return call

            def _emit_abs_max_ops_chain(self, expr: relax.Var) -> relax.Var:
                assert expr.struct_info.ndim >= 2, "Tensor dim num should be >= 2"
                abs_expr = self.builder_.emit(R.abs(expr))
                max_expr = self.builder_.emit(R.max(abs_expr, axis=-2))
                if expr.struct_info.ndim > 2:
                    max_expr = self.builder_.emit(R.squeeze(max_expr))
                return max_expr

            def _emit_max_min_ops_chain(self, expr: relax.Var, axis: int) -> List[relax.Var]:
                assert expr.struct_info.ndim >= 2, "Tensor dim num should be >= 2"
                max_expr = self.builder_.emit(R.max(expr, axis=axis))
                min_expr = self.builder_.emit(R.min(expr, axis=axis))
                if expr.struct_info.ndim > 2:
                    max_expr = self.builder_.emit(R.squeeze(max_expr))
                    min_expr = self.builder_.emit(R.squeeze(min_expr))
                return max_expr, min_expr

        return ParamsAndOutputsMutator(mod).transform()


def is_zero(expr: Union[int, tvm.relax.Constant]) -> bool:
    if isinstance(expr, int) and expr == 0:
        return True
    return False


@tvm.transform.module_pass(opt_level=0, name="SmoothQuantLegalizer")
class SmoothQuantLegalizer:
    """
    Pass that performs the following transformation:
    quantize + dequantize + matmul(fp16, fp16) -> quantize + matmul(int8, int8) + dequantize.
    """
    def __init__(self, adtype: str = "int8", wdtype: str = "int8"):
        self.dtype_act = adtype
        self.dtype_weight = wdtype

    def transform_module(self, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.IRModule:
        act = wildcard()
        act_scale = wildcard()
        act_zp = wildcard()
        weights = wildcard()
        w_scale = wildcard()
        w_zp = wildcard()
        lhs_sm = is_op("relax.dequantize")(act, act_scale, act_zp)
        rhs_sm = is_op("relax.dequantize")(weights, w_scale, w_zp)
        permute = is_op("relax.permute_dims")(rhs_sm)
        pattern = is_op("relax.matmul")(lhs_sm, permute)

        def rewriter(_, matchings):
            reduction_axis_size: int = matchings[lhs_sm].struct_info.shape[-1].value
            dtype = matchings[pattern].struct_info.dtype
            mm_shape = matchings[pattern].struct_info.shape

            def _simplify_constant(expr: tvm.relax.Expr) -> tvm.relax.Expr:
                """
                Simplify R.const([value, value ... value]) -> R.const([value])
                """
                if not isinstance(expr, tvm.relax.Constant):
                    return expr
                if expr.struct_info.ndim == 1 and expr.struct_info.shape[0].value > 1:
                    # check that all elements of array are the same.
                    if np.all(np.isclose(expr.data.numpy(), expr.data.numpy()[0])):
                        scalar = expr.data.numpy()[0].item()
                        return R.const([scalar], expr.struct_info.dtype)
                return expr

            def _make_dequantize(
                call: tvm.relax.Call,
                scale1: tvm.relax.Constant,
                scale2: tvm.relax.Constant,
                axis: int,
                out_dtype: str,
            ):
                scalar_scale1 = _try_convert_to_scalar_const(scale1)
                scalar_scale2 = _try_convert_to_scalar_const(scale2)
                if isinstance(scalar_scale1, float) and isinstance(scalar_scale2, float):
                    size = mm_shape[axis].value
                    scale = R.const([scalar_scale1 * scalar_scale2] * size, "float32")
                else:
                    scale = R.multiply(R.astype(scale1, "float32"), R.astype(scale2, "float32"))
                return R.dequantize(call, scale, R.const(0, "int8"), axis=axis, out_dtype=out_dtype)

            def _make_linear(
                lhs: tvm.relax.Call,
                rhs: tvm.relax.Call,
                lhs_zp: tvm.relax.Constant,
                rhs_zp: tvm.relax.Constant,
                reduction_dim_size: int
            ):
                lhs_zp_const = _try_convert_to_scalar_const(lhs_zp)
                rhs_zp_const = _try_convert_to_scalar_const(rhs_zp)
                # Make term1:
                term1 = R.linear(lhs, rhs, out_dtype="int32")
                # Make term2:
                lhs_reduced = R.sum(R.astype(lhs, dtype="int32"), axis=-1, keepdims=True)
                rhs_zp = R.astype(rhs_zp, dtype="int32")
                term2 = R.multiply(lhs_reduced, rhs_zp)
                # Make term3:
                rhs_reduced = R.sum(R.astype(rhs, dtype="int32"), axis=-1, keepdims=False)
                lhs_zp = R.astype(lhs_zp, dtype="int32")
                term3 = R.multiply(rhs_reduced, lhs_zp)
                # Make term4:
                term4 = R.multiply(R.multiply(lhs_zp, rhs_zp), R.const(reduction_dim_size, "int32"))
                # Combine result:
                if is_zero(lhs_zp_const) and is_zero(rhs_zp_const):
                    return term1
                elif is_zero(lhs_zp_const) and not is_zero(rhs_zp_const):
                    return R.subtract(term1, term2)
                elif not is_zero(lhs_zp_const) and is_zero(rhs_zp_const):
                    return R.subtract(term1, term3)
                else:
                    return R.add(R.subtract(R.subtract(term1, term2), term3), term4)

            lhs = matchings[act]
            rhs = matchings[weights]
            scale1 = _simplify_constant(matchings[act_scale])
            scale2 = _simplify_constant(matchings[w_scale])
            lhs_zp = _simplify_constant(matchings[act_zp])
            rhs_zp = _simplify_constant(matchings[w_zp])
            mm = _make_linear(lhs, rhs, lhs_zp, rhs_zp, reduction_axis_size)
            return _make_dequantize(mm, scale1, scale2, axis=-1, out_dtype=dtype)


        new_mod = tvm.IRModule()
        for gv, func in mod.functions.items():
            if isinstance(func, relax.Function):
                new_mod[gv] = rewrite_call(pattern, rewriter, func)
            else:
                new_mod[gv] = func
        return new_mod


@tvm.relax.transform.function_pass(opt_level=0, name="SmoothQuantStopLiftParamsOptimizer")
class SmoothQuantStopLiftParamsOptimizer:
    """
    Transformation pass to insert stop_lift_params op before linear op.
    """

    def __init__(self):
        self.input = wildcard()
        self.weights = wildcard()
        permute = is_op("relax.permute_dims")(self.weights)
        self.pattern = is_op("relax.matmul")(self.input, permute)

    def transform_function(self, func, mod: tvm.IRModule, ctx: tvm.transform.PassContext):
        if not isinstance(func, relax.Function):
            return func

        def rewriter(expr, matches):
            weights = matches[self.weights]
            if weights.struct_info.ndim != 2:
                return expr
            if isinstance(weights, relax.Call) and weights.op == tvm.ir.Op.get("relax.builtin.stop_lift_params"):
                return expr
            stop =  R.builtin.stop_lift_params(weights)
            out_dtype = matches[self.pattern].attrs.out_dtype
            return R.linear(matches[self.input], stop, out_dtype=out_dtype)

        return rewrite_call(self.pattern, rewriter, func)


@tvm.relax.transform.function_pass(opt_level=0, name="SmoothQuantParamsMutator")
class SmoothQuantParamsMutator:
    """
    Transformation pass to substitute param with quantized one and remove the following sequence of
    operations: R.quantize(R.multiply(...), ...).
    """

    def __init__(self, dtype):
        self.new_dtype = dtype

        self.weights = is_tuple_get_item(wildcard())
        multiply = is_op("relax.multiply")(self.weights, wildcard())
        quantize = is_op("relax.quantize")(multiply, wildcard(), wildcard())
        self.pattern = is_op("relax.dequantize")(quantize, wildcard(), wildcard()) | is_tuple_get_item(wildcard())

    def transform_function(self, func, mod: tvm.IRModule, ctx: tvm.transform.PassContext):
        if not isinstance(func, relax.Function):
            return func

        @visitor
        class TupleGetItemVisitor(PyExprVisitor):
            def __init__(self):
                self.indexes = set()
                self.get_item = wildcard()
                multiply = is_op("relax.multiply")(self.get_item, wildcard())
                quantize = is_op("relax.quantize")(multiply, wildcard(), wildcard())
                self.pattern = is_op("relax.dequantize")(quantize, wildcard(), wildcard())

            def collect(self, func):
                self.var2val = tvm.relax.analysis.get_var2val(func)
                self.visit_expr(func)
                return self.indexes

            def visit_call_(self, call: tvm.relax.Call) -> None:
                matchings = self.pattern.extract_matched_expr(call, self.var2val)
                if matchings:
                    tgi = matchings[self.get_item]
                    assert isinstance(tgi, relax.TupleGetItem)
                    self.indexes.add(tgi.index)
                super().visit_call_(call)


        indexes = TupleGetItemVisitor().collect(func)

        num_input = int(func.attrs["num_input"])
        params = func.params[num_input:]
        sinfo = []
        for param in params:
            if isinstance(param.struct_info, relax.TupleStructInfo):
                self.tuple_name = param.name_hint
                for idx, field in enumerate(param.struct_info.fields):
                    dtype = self.new_dtype if idx in indexes else field.dtype
                    sinfo.append(relax.TensorStructInfo(field.shape, dtype))
        self.var_param_tuple = relax.Var("packed_params", relax.TupleStructInfo(sinfo))

        def rewriter(expr, matches):
            op = matches[self.pattern]
            if isinstance(op, relax.TupleGetItem):
                if op.index not in indexes and op.tuple_value.name_hint == self.tuple_name:
                    return relax.TupleGetItem(self.var_param_tuple, op.index)
                else:
                    return expr
            weights = matches[self.weights]
            dq = op
            axis = dq.attrs.axis
            out_dtype = dq.attrs.out_dtype
            tgi = relax.TupleGetItem(self.var_param_tuple, weights.index)
            return R.dequantize(tgi, dq.args[1], dq.args[2], axis=axis, out_dtype=out_dtype)

        new_func = rewrite_call(self.pattern, rewriter, func)
        return relax.Function(
            params=func.params[:num_input] + [self.var_param_tuple],
            body=new_func.body,
            ret_struct_info=new_func.ret_struct_info,
            is_pure=new_func.is_pure,
            attrs=new_func.attrs,
        )


@mutator
class ParamBundler(PyExprMutator):
    def __init__(self, mod: tvm.IRModule) -> None:
        super().__init__(mod)
        self.mod = mod
        self.var_to_expr = dict()

    def transform(self) -> tvm.IRModule:
        for gv, func in self.mod.functions.items():
            if not isinstance(func, relax.Function):
                continue
            updated_func = self.visit_expr(func)
            updated_func = remove_all_unused(updated_func)
            self.builder_.update_func(gv, updated_func)

        return self.builder_.get()

    def visit_function_(self, f) -> tvm.relax.Function:
        num_input = int(f.attrs["num_input"])
        params = f.params[num_input:]
        sinfo = []
        for param in params:
            if isinstance(param.struct_info, relax.TupleStructInfo):
                sinfo += param.struct_info.fields
            else:
                assert isinstance(param.struct_info, relax.TensorStructInfo)
                sinfo.append(param.struct_info)
        var_param_tuple = relax.Var("packed_params", relax.TupleStructInfo(sinfo))
        idx = 0
        for param in params:
            if isinstance(param.struct_info, relax.TupleStructInfo):
                self.var_to_expr[param] = (idx, var_param_tuple)
                idx += len(param.struct_info.fields)
            else:
                assert isinstance(param.struct_info, relax.TensorStructInfo)
                self.var_to_expr[param] = relax.TupleGetItem(var_param_tuple, idx)
                idx += 1
        body = super().visit_expr(f.body)
        return relax.Function(
            params=f.params[:num_input] + [var_param_tuple],
            body=body,
            ret_struct_info=f.ret_struct_info,
            is_pure=f.is_pure,
            attrs=f.attrs,
            span=f.span,
        )

    def visit_tuple_getitem_(self, tgi):
        if tgi.tuple_value in self.var_to_expr:
            start_idx, var = self.var_to_expr[tgi.tuple_value]
            return relax.TupleGetItem(var, start_idx + tgi.index)
        return super().visit_tuple_getitem_(tgi)

    def visit_var_(self, var) -> tvm.relax.Expr:
        if var in self.var_to_expr:
            return self.var_to_expr[var]
        return super().visit_var_(var)


@tvm.transform.module_pass(opt_level=0, name="SmoothQuantBundleModelParams")
class SmoothQuantBundleModelParams:
    def transform_module(self, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.IRModule:
        return ParamBundler(mod).transform()

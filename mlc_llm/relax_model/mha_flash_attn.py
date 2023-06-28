# Implementation of flash attention (multi-head attention) on Relax based on https://github.com/HazyResearch/flash-attention
from tvm import relax
from tvm.relax.testing import nn

from einops import rearrange  # TODO: remove


def index_first_axis(input, indices):
  assert input.struct_info.ndim >= 2
  batch_size, *other_shape = input.struct_info.shape
  second_dim = relax.const(1)
  for dim in other_shape:
    second_dim = second_dim * dim
  # For some reason gather is a bit faster than indexing.
  # return input[indices]
  return nn.emit(relax.op.reshape(
    # TODO: gather on Relax
    torch.gather(
      relax.op.reshape(input, (batch_size, -1))   # b ... -> b (...)
      0,
      relax.op.repeat(indices, second_dim, 1)     # z -> z d
    ),
    (-1, *other_shape),
  ))


def index_put_first_axis(values, indices, first_axis_dim):
  assert indices.struct_info.ndim == 1
  assert values.struct_info.ndim >= 2
  _, *other_shape = values.struct_info.shape
  repeats = values.struct_info.shape[1]
  output = nn.emit(relax.op.zeros((first_axis_dim, *other_shape), values.struct_info.dtype))
  output = nn.emit(relax.op.scatter_elements(
    output,
    relax.op.repeat(indices, repeats, 1),  # z -> z d
    values))
  return output


def bert_padding_unpad_input(hidden_states, attention_mask):
  seqlens_in_batch = nn.emit(relax.op.sum(attention_mask, -1))
  max_seqlen_in_batch = relax.op.max(seqlens_in_batch)
  # TODO: implement nonzero or argwhere on relax and replace torch one
  # TODO: possibly the first flatten is excess
  indices = nn.emit(relax.op.flatten(
    torch.nonzero(
      relax.op.flatten(attention_mask),
      as_tuple=False
    )
  ))
  # TODO: implement pad on Relax and replace torch one
  cu_seqlens = nn.emit(torch.nn.functional.pad(
    relax.op.cumsum(seqlens_in_batch, 0 , dtype="int32"),
    (1, 0)
  ))
  batch_size = hidden_states.struct_info.shape[0]
  seq_len = hidden_states.struct_info.shape[1]
  return (
      index_first_axis(
        relax.op.reshape(hidden_states, (batch_size * seq_len, -1)), # b s ... -> (b s) ...
        indices
      ),
      indices,
      cu_seqlens,
      max_seqlen_in_batch
  )


def bert_padding_pad_input(hidden_states, indices, batch, seqlen):
  """
  Arguments:
      hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
      indices: (total_nnz)
  Return:
      hidden_states: (batch, seqlen, ...)
  """
  output = index_put_first_axis(hidden_states, indices, batch * seqlen)
  return nn.emit(relax.op.reshape(output, (batch, seqlen, -1))) # (b s) ... -> b s ...


def _flash_attn_forward(q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                        dropout_p, softmax_scale, causal, return_softmax, num_splits=0,
                        generator=None):
    """
    num_splits: how much to parallelize over the seqlen_q dimension. num_splits=0 means
    it will be set by an internal heuristic. We're exposing num_splits mostly for benchmarking.
    Don't change it unless you know what you're doing.
    """
    # CUDA kernel is implemented in mha_frw method from https://github.com/HazyResearch/flash-attention/blob/main/csrc/flash_attn/fmha_api.cpp
    # TODO(vchernov): CUDA kernel should be replaced by implementation on TVM side
    # import flash_attn_cuda
    # softmax_lse, rng_state, *rest = flash_attn_cuda.fwd(
    #     q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, 0, # dropout_p = 0 for inference
    #     softmax_scale, False, causal, return_softmax, num_splits, generator
    # )
    # S_dmask = rest[0] if return_softmax else None
    out = None          # Dummy value while computation is commented
    softmax_lse = None  # Dummy value while computation is commented
    rng_state = None    # Dummy value while computation is commented
    S_dmask = None      # Dummy value while computation is commented
    return out, softmax_lse, rng_state, S_dmask


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
                softmax_scale, causal, return_softmax, deterministic):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse, rng_state, S_dmask = _flash_attn_forward(
            q, k, v, torch.empty_like(q), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p, softmax_scale, causal=causal, return_softmax=return_softmax
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
        ctx.dropout_p = dropout_p
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)


def flash_attn_unpadded_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
    dropout_p, softmax_scale=None, causal=False, return_attn_probs=False,
    deterministic=False
):
  return FlashAttnFunc.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                              dropout_p, softmax_scale, causal, return_attn_probs, deterministic)

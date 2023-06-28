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


import torch
class IndexPutFirstAxis(torch.autograd.Function):

    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device,
                             dtype=values.dtype)
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output


index_put_first_axis = IndexPutFirstAxis.apply


def bert_padding_unpad_input(hidden_states, attention_mask):
  seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
  indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
  max_seqlen_in_batch = seqlens_in_batch.max().item()
  cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
  # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
  # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
  # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
  # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
  # so we write custom forward and backward to make it a bit faster.
  return (index_first_axis(rearrange(hidden_states, 'b s ... -> (b s) ...'), indices), indices,
          cu_seqlens, max_seqlen_in_batch)


def bert_padding_pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz)
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[-1]
    # output = torch.zeros((batch * seqlen), dim, device=hidden_states.device, dtype=hidden_states.dtype)
    # output[indices] = hidden_states
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, '(b s) ... -> b s ...', b=batch)


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

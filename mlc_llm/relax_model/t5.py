import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tvm
from tvm import relax, te
from tvm.relax.testing import nn
from tvm.script import relax as R
from .llama import LlamaRMSNorm, Linear, Embedding
# from .modules import (
#     Linear
# )

@dataclass
class T5Config:
    def __init__(
        self,
        d_kv,
        d_model,
        d_ff,
        num_heads,
        num_layers,
        layer_norm_epsilon,
        relative_attention_num_buckets,
        relative_attention_max_distance,
        feed_forward_proj,
        num_decoder_layers,
        tie_word_embeddings,
        dtype="float32",
        vocab_size=32128,
        **kwargs,
    ):
        self.dtype = dtype
        self.d_kv = d_kv
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers
        self.layer_norm_epsilon = layer_norm_epsilon

        self.is_decoder = True
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance

        self.output_attentions = False # is absent in t5 base config, but PretrainedConfig default is false
        self.output_hidden_states = False # is absent in t5 base config, but PretrainedConfig default is false

        act_info = feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        self.vocab_size = vocab_size
        self.use_cache = False
        self.tie_word_embeddings = tie_word_embeddings
        self.kwargs = kwargs


T5LayerNorm = LlamaRMSNorm

class T5Utils:
    # emulates torch.arange behaviour
    @staticmethod
    def arange(len):
        return te.compute(
            (len,),
            lambda i: i.astype("int32"),
            name="arange",
            )


class T5DenseActDense(nn.Module):
    def __init__(self, config: T5Config):
        self.wi = Linear(config.d_model, config.d_ff, dtype = config.dtype, bias=False)
        self.wo = Linear(config.d_ff, config.d_model, dtype = config.dtype, bias=False)
        self.act = config.dense_act_fn

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):
    def __init__(self, config: T5Config):
        self.wi_0 = Linear(config.d_model, config.d_ff, dtype = config.dtype, bias=False)
        self.wi_1 = Linear(config.d_model, config.d_ff, dtype = config.dtype, bias=False)
        self.wo = Linear(config.d_ff, config.d_model, dtype = "float32", bias=False)
        self.act = config.dense_act_fn
        self.dtype = config.dtype

    def forward(self, hidden_states):
        from tvm.relax.op.nn import gelu
        from tvm.relax.op import astype

        if self.act == 'gelu':
            hidden_gelu = gelu(self.wi_0(hidden_states))
        else:
            assert 0
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        if self.dtype != "float32":
            hidden_states = astype(hidden_states, "float32")
        hidden_states = self.wo(hidden_states)
        if self.dtype != "float32":
            hidden_states = astype(hidden_states, self.dtype)
        return hidden_states


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, dtype = config.dtype, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = nn.emit(hidden_states + forwarded_states)
        return hidden_states


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = Linear(self.d_model, self.inner_dim, dtype = config.dtype, bias=False)
        self.k = Linear(self.d_model, self.inner_dim, dtype = config.dtype, bias=False)
        self.v = Linear(self.d_model, self.inner_dim, dtype = config.dtype, bias=False)
        self.o = Linear(self.inner_dim, self.d_model, dtype = config.dtype, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = Embedding(self.relative_attention_num_buckets, self.n_heads, dtype=config.dtype)
        self.pruned_heads = set()
        self.gradient_checkpointing = False
        self.config = config

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """

        # emulating relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
        def mult_positive(x, num_buckets):
            return te.compute(
                x.shape,
                lambda i, j: te.if_then_else(
                    x[i, j]  > 0,
                    num_buckets,
                    0,
                ),
                name="mult_positive",
            )

        # relative_position_if_large = max_exact + (
        #     torch.log(relative_position.float() / max_exact)
        #     / math.log(max_distance / max_exact)
        #     * (num_buckets - max_exact)
        # ).to(torch.long)
        # relative_position_if_large = torch.min(
        #     relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        # )
        def relative_position_if_large_calc(x, max_exact, max_distance, num_buckets):
            lde = math.log(max_distance / max_exact)
            return te.compute(
                x.shape,
                lambda i, j: tvm.tir.min(
                    max_exact + (tvm.tir.log(x[i,j].astype("float32") / max_exact) / lde * (num_buckets - max_exact)).astype("int32"),
                    num_buckets - 1)
                ,
                name="relative_position_if_large_calc",
            )

        # is_small = relative_position < max_exact
        # relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        def relative_buckets_calc(x, y, max_exact):
            return te.compute(
                x.shape,
                lambda i, j: te.if_then_else(
                    x[i, j]  < max_exact,
                    x[i, j],
                    y[i, j],
                ),
                name="relative_buckets",
            )
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets = nn.emit_te(mult_positive, relative_position, num_buckets)
            relative_position = nn.emit(relax.op.abs(relative_position))
        else:
            def torch_min_mult(x, y, dtype):
                return te.compute(
                    x.shape,
                    lambda i, j: -1 * tvm.tir.min(
                        x[i,j],
                        y[i,j]).astype(dtype)
                    ,
                    name="relative_position_if_large_calc",
                )
            relative_position = nn.emit_te(torch_min_mult,
                                           relative_position,
                                           nn.emit(relax.op.zeros_like(relative_position, dtype = relative_position.struct_info.dtype)),
                                           relative_position.struct_info.dtype
            )
            relative_buckets = nn.emit(relax.op.zeros_like(relative_position, dtype = "int32"))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = nn.emit_te(relative_position_if_large_calc, relative_position, max_exact, max_distance, num_buckets)

        relative_buckets = nn.emit( relative_buckets +
            nn.emit_te(relative_buckets_calc, relative_position, relative_position_if_large, max_exact))
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        from tvm.relax.op import permute_dims, expand_dims, reshape

        context_position = nn.emit(reshape(nn.emit_te(T5Utils.arange, query_length), (query_length, 1)))
        memory_position = nn.emit(reshape(nn.emit_te(T5Utils.arange, key_length), (1, key_length)))
        relative_position = nn.emit(memory_position - context_position)  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )

        #values = self.relative_attention_bias(nn.emit(astype(relative_position_bucket, self.config.dtype)))  # shape (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = nn.emit(expand_dims(permute_dims(values, [2, 0, 1]), axis=[0]))
        #values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        from tvm.relax.op import astype, matmul, maximum, permute_dims, reshape, squeeze
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length, _ = hidden_states.struct_info.shape

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.struct_info.shape[1]

        def shape(states):
            """projection"""
            return nn.emit(
                permute_dims(
                    reshape(states, (batch_size, -1, self.n_heads, self.key_value_proj_dim)),
                    [0, 2, 1, 3]
                )
            )

        def unshape(states):
            """reshape"""
            pd = nn.emit(permute_dims(states, [0, 2, 1, 3]))
            return nn.emit(
                reshape(
                    pd,
                    [batch_size, -1, self.inner_dim]
                    )
            )

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = nn.emit(
            matmul(query_states, permute_dims(key_states, [0, 1, 3, 2]))
        )

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = nn.emit(relax.op.zeros((1, self.n_heads, real_seq_length, key_length), scores.struct_info.dtype))
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                #position_bias = position_bias[:, :, -hidden_states.size(1) :, :]
                position_bias = nn.emit(relax.op.split(position_bias, [position_bias.struct_info.shape[2] - hidden_states.struct_info.shape[1]], axis = 2))

            if mask is not None:
                position_bias = nn.emit(position_bias + mask)  # (batch_size, n_heads, seq_length, key_length)

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        scores = nn.emit(scores + nn.emit(position_bias_masked))
        scores_dtype = scores.struct_info.dtype
        if scores_dtype != "float32":
            scores = astype(scores, "float32")
        attn_weights = nn.emit(relax.op.nn.softmax(scores, axis=-1))
        if attn_weights.struct_info.dtype != scores_dtype:
            attn_weights = astype(attn_weights, scores_dtype)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(nn.emit(matmul(attn_weights, value_states)))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, dtype=config.dtype, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = nn.emit(hidden_states + attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, dtype = config.dtype, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = nn.emit(hidden_states + attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = []
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        if past_key_value is not None:
            # TODO(amalyshe): this branch should be executed in decoder
            assert 0
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        # if hidden_states.dtype == torch.float16:
        #     clamp_value = torch.where(
        #         torch.isinf(hidden_states).any(),
        #         torch.finfo(hidden_states.dtype).max - 1000,
        #         torch.finfo(hidden_states.dtype).max,
        #     )
        #     hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].struct_info.shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # it was a clamp for fp16 training, here, let's try to apply fp16 clamping for inference
            # in pure fp16
            if hidden_states.struct_info.dtype == "float16":
                hidden_states = nn.emit(relax.op.clip(hidden_states, -64504., 64504.))

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](nn.emit(hidden_states))

        # clamp inf values to enable fp16 training
        # if hidden_states.dtype == torch.float16:
        #     clamp_value = torch.where(
        #         torch.isinf(hidden_states).any(),
        #         torch.finfo(hidden_states.dtype).max - 1000,
        #         torch.finfo(hidden_states.dtype).max,
        #     )
        #     hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        if hidden_states.struct_info.dtype == "float16":
            hidden_states = nn.emit(relax.op.clip(hidden_states, -64504., 64504.))
        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)

class T5Stack(nn.Module):
    def __init__(self, config, shared_embed_tokens: Embedding):
        self.is_decoder = config.is_decoder

        self.embed_tokens = shared_embed_tokens
        self.block = [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        self.final_layer_norm = T5LayerNorm(config.d_model, dtype = config.dtype, eps=config.layer_norm_epsilon)
        self.config = config

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def create_extended_attention_mask_for_decoder(self, input_shape, attention_mask, device=None):
        from tvm.relax.op import reshape
        batch_size, seq_length = input_shape
        if batch_size == 1 and seq_length == 1:
           return nn.emit(reshape(attention_mask, (1,1,1,1)))
        else:
           assert 0
           #below code should be implemented on relax
        # batch_size, seq_length = input_shape.struct_info.shape
        # seq_ids = nn.emit(reshape(nn.emit_te(T5Utils.arange, seq_length), (1, 1, seq_length)))
        # causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        # # causal and attention masks must have same type with pytorch version < 1.3
        # causal_mask = causal_mask.to(attention_mask.dtype)

        # if causal_mask.shape[1] < attention_mask.shape[1]:
        #     prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
        #     causal_mask = torch.cat(
        #         [
        #             torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
        #             causal_mask,
        #         ],
        #         axis=-1,
        #     )

        # extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        # return extended_attention_mask


    def get_extended_attention_mask(
        self, attention_mask, input_shape, dtype = "float32"
    ):
        from tvm.relax.op import astype, broadcast_to
        batch_size = None
        seq_length = None
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.struct_info.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.struct_info.ndim == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            batch_size, seq_length = attention_mask.struct_info.shape
            if self.config.is_decoder:
                extended_attention_mask = self.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask)
            else:
                extended_attention_mask = nn.emit(broadcast_to(attention_mask, (batch_size, 1, 1, seq_length)))
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        att_mask_filled_min = nn.emit(
            relax.op.full(
                (batch_size, 1, 1, seq_length),
                relax.const(tvm.tir.min_value(dtype).value, dtype),
                dtype,
            )
        )
        extended_attention_mask = nn.emit(
            (relax.const(1.0, dtype) - astype(extended_attention_mask, dtype)) * att_mask_filled_min)

        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask):
        """
        Implement the same behaviour as in Transformer's ModuleUtilsMixin.invert_attention_mask
        """
        from tvm.relax.op import astype, broadcast_to
        dtype = encoder_attention_mask.struct_info.dtype
        if encoder_attention_mask.struct_info.ndim == 3:
            #TODO(amalyshe): need to remove?
            assert 0
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            encoder_extended_attention_mask = nn.emit(broadcast_to(encoder_attention_mask, (batch_size, 1, 1, seq_length)))
        if encoder_attention_mask.struct_info.ndim == 2:
            batch_size, seq_length = encoder_attention_mask.struct_info.shape
            encoder_extended_attention_mask = nn.emit(broadcast_to(encoder_attention_mask, (batch_size, 1, 1, seq_length)))
        mask_filled_min = nn.emit(
            relax.op.full(
                (batch_size, 1, 1, seq_length),
                relax.const(tvm.tir.min_value(dtype).value, dtype),
                dtype,
            )
        )
        encoder_extended_attention_mask = nn.emit(
            astype((relax.const(1.0, dtype) - astype(encoder_extended_attention_mask, dtype)) * mask_filled_min,
                    self.config.dtype))

        return encoder_extended_attention_mask

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            assert ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.struct_info.shape
            # input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            assert ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].struct_info.shape[2] + seq_length if past_key_values is not None else seq_length

        # if use_cache is True:
        #     assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = nn.emit(relax.op.ones((batch_size, mask_seq_length), self.config.dtype))

        # if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
        #     encoder_seq_length = encoder_hidden_states.shape[1]
        #     encoder_attention_mask = torch.ones(
        #         batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
        #     )

        # # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask,
                                                                   input_ids.struct_info.shape,
                                                                   dtype = self.config.dtype)

        # # If a 2D or 3D attention mask is provided for the cross-attention
        # # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.struct_info.shape
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = nn.emit(relax.op.ones(encoder_hidden_shape, self.config.dtype))
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # if self.gradient_checkpointing and self.training:
        #     if use_cache:
        #         logger.warning_once(
        #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
        #         )
        #         use_cache = False

        # # Prepare head mask if needed
        # TODO(amalyshe): it was an invocation of get_head_mask which extends head_masks passed
        # through params of this forward func. But we assume so far that head_masks is None all the time
        # same for cross_attn_head_mask
        head_mask = [None] * self.config.num_layers
        cross_attn_head_mask = [None] * self.config.num_layers
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = inputs_embeds

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # TODO(amalyshe): assuming that output_hidden_states is always False
            # if output_hidden_states:
            #     all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = nn.emit(self.final_layer_norm(hidden_states))

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                present_key_value_states,
                all_hidden_states,
                all_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )

def create_encoding_func(bb: relax.BlockBuilder, config: T5Config) -> None:
    bsz = 1
    seq_len = tvm.tir.Var("n", "int64")
    with bb.function("encode"):
        config.is_decoder = False
        config.use_cache = False

        # TODO(amalyshe): below embeddings are shared between encoder and decoder, originally it was declared in
        # T5ForConditionalGeneration but I started to implement from encoder only
        # and we do not have T5ForConditionalGeneration so far or may be never have it
        # but we need to create shared embed tokens. Here they are
        # and due to restrictions of nn.Params, this creation should be called inside
        # with bb.function...:
        # that is conflicts with nature of shared param for encoder and decoder
        shared_embed_tokens = Embedding(config.vocab_size, config.d_model, dtype=config.dtype)

        model = T5Stack(config, shared_embed_tokens)
        input_ids = nn.Placeholder((bsz, seq_len), dtype="int32", name="input_ids")
        attention_mask = nn.Placeholder((bsz, seq_len), dtype="int32", name="attention_mask")
        with bb.dataflow():
            last_hidden_state = model(
                input_ids = input_ids, attention_mask = attention_mask
            )
            params = [
                input_ids,
                attention_mask,
            ] + model.parameters()
            gv = bb.emit_output(last_hidden_state)
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var("encode")
    bb.update_func(gv, mod[gv].with_attr("num_input", 2))


class T5ForConditionalGenerationRelax(nn.Module):
    def __init__(self, config: T5Config, shared_embed_tokens: Embedding):
        self.config = config
        self.model_dim = config.d_model
        # self.shared = Embedding(config.vocab_size, config.d_model, dtype = config.dtype)

        decoder_config = config
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        #self.decoder = T5Stack(decoder_config, self.shared)
        self.decoder = T5Stack(decoder_config, shared_embed_tokens)

        self.lm_head = Linear(config.d_model, config.vocab_size, dtype = config.dtype, bias=False)

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        decoder_input_ids = None,
        decoder_attention_mask = None,
        head_mask = None,
        decoder_head_mask = None,
        cross_attn_head_mask = None,
        encoder_outputs = None,
        past_key_values = None,
        inputs_embeds = None,
        decoder_inputs_embeds = None,
        labels = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        #return (lm_logits,)
        output = (lm_logits,)
        for i in decoder_outputs[1:][0]:
            for j in i:
                output += (j,)
        return output

        #return (lm_logits,) + decoder_outputs[1:]
        # if not return_dict:
        #     output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        #     return ((loss,) + output) if loss is not None else output

        # return Seq2SeqLMOutput(
        #     loss=loss,
        #     logits=lm_logits,
        #     past_key_values=decoder_outputs.past_key_values,
        #     decoder_hidden_states=decoder_outputs.hidden_states,
        #     decoder_attentions=decoder_outputs.attentions,
        #     cross_attentions=decoder_outputs.cross_attentions,
        #     encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        #     encoder_hidden_states=encoder_outputs.hidden_states,
        #     encoder_attentions=encoder_outputs.attentions,
        # )



def create_decoding_first_iter_func(bb: relax.BlockBuilder, config: T5Config) -> None:
    bsz = 1
    seq_len = tvm.tir.Var("n", "int64")
    with bb.function("decode"):
        config.is_decoder = True
        config.use_cache = True
        shared_embed_tokens = Embedding(config.vocab_size, config.d_model, dtype=config.dtype)
        model = T5ForConditionalGenerationRelax(config, shared_embed_tokens)
        decoder_input_ids = nn.Placeholder((bsz, 1), dtype="int32", name="decoder_input_ids")
        # 768 must be a multiplications from config
        # TODO(amalyshe) - verify if config.num_heads * config.d_kv is correct
        encoder_outputs = nn.Placeholder((bsz, seq_len, config.num_heads * config.d_kv),
                                         dtype=config.dtype, name="decoder_input_ids")
        attention_mask = nn.Placeholder((bsz, seq_len), dtype="int32", name="attention_mask")
        output_attentions = False
        output_hidden_states = False

        with bb.dataflow():
            lm_logits = model(
                decoder_input_ids = decoder_input_ids,
                attention_mask = attention_mask,
                encoder_outputs = encoder_outputs,
                output_attentions = output_attentions,
                output_hidden_states = output_hidden_states,
            )
            params = [
                decoder_input_ids,
                attention_mask,
                encoder_outputs,
            ] + model.parameters()
            #gv = bb.emit_output(relax.Tuple(last_hidden_state))
            gv = bb.emit_output(lm_logits)
        bb.emit_func_output(gv, params)

    mod = bb.get()
    gv = mod.get_global_var("decode")
    bb.update_func(gv, mod[gv].with_attr("num_input", 3))


def get_model(args, hf_config):
    from transformers import T5ForConditionalGeneration # type: ignore[import]

    model_name = args.model
    model_path = args.model_path
    dtype = args.quantization.model_dtype
    max_seq_len = args.max_seq_len

    if model_name.startswith("flan-t5-"):
        config = T5Config(**hf_config, dtype=dtype)
        # if max_seq_len != -1:
        #     config.max_sequence_length = max_seq_len

        bb = relax.BlockBuilder()
        create_encoding_func(bb, config)
        create_decoding_first_iter_func(bb, config)
        mod = bb.get()

        device = tvm.cpu()
        hf_model = T5ForConditionalGeneration.from_pretrained(model_path)
        # Get a list of parameters in advance, then delete the model to save memory
        # param_list = [param for _, param in hf_model.named_parameters()]
        # i = 0
        # for name, param in hf_model.named_parameters():
        #     print(name, param.shape)
        #     if i == 21:
        #         print(name, param.shape, param)
        #     i = i + 1
        # Get a list of parameters in advance, then delete the model to save memory

        params2 = {}
        param_list = []
        for name, param in hf_model.named_parameters():
            tdtype = "float32" if "wo.weight" in name else config.dtype
            param_list.append(tvm.nd.array(
                    param.detach().cpu().numpy().astype(tdtype), device
                )
            )

        # param_list = [param for _, param in hf_model.named_parameters()]

        # for i, param in enumerate(param_list):
        #     param_list[i] = tvm.nd.array(
        #         param.detach().cpu().numpy().astype(config.dtype), device
        #     )
        # del hf_model

        print(mod)
        return mod, param_list

    raise ValueError(f"Unsupported model: {model_name}")

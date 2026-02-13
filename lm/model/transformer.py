from __future__ import annotations

from enum import StrEnum, auto
from functools import partial

import equinox as eqx
import jax
import jax.ad_checkpoint
import jax.nn
import jax.numpy as jnp
import jax.random as jrandom
from equinox import nn
from jaxtyping import PRNGKeyArray

from lm.config import Config, ModelConfig
from lm.model.attention import Attention, AttentionBase, NormalLinear, SlidingWindowAttention
from lm.model.data import BaseModelOutput, Batch
from lm.model.loss import cross_entropy_loss_and_accuracy, token_log_probs
from lm.optimizers import make_optimizer
from lm.utils.filter_utils import filter_apply_updates, filter_parameters, get_filter_spec
from lm.utils.jax_utils import (
    get_float_dtype_by_name,
    maybe_double_remat,
    promote_dtype,
    scan_or_loop,
    tree_rearrange,
)


class SwiGLUMLP(eqx.Module):
    """Single SwiGLU MLP block"""

    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)
    w1: NormalLinear
    w2: NormalLinear
    w3: NormalLinear
    dropout: nn.Dropout = eqx.field(static=True)

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.param_dtype)

        w1_key, w2_key, w3_key = jrandom.split(key, 3)

        self.w1 = NormalLinear(
            self.config, in_features=config.hidden_size, out_features=config.intermediate_size, std=config.initializer_range, key=w1_key, name="w1"
        )

        self.w2 = NormalLinear(
            self.config, in_features=config.intermediate_size, out_features=config.hidden_size, std=config.initializer_range, key=w2_key, name="w2"
        )

        self.w3 = NormalLinear(
            self.config, in_features=config.hidden_size, out_features=config.intermediate_size, std=config.initializer_range, key=w3_key, name="w3"
        )

        self.dropout = nn.Dropout(p=self.config.resid_pdrop)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z1 = self.w1(x)
        z1_act = jax.nn.silu(z1)
        z3 = self.w3(x)
        x2 = z1_act * z3
        z2 = self.w2(x2)
        output = self.dropout(z2)
        return output


class Block(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    seq_modeling_block: AttentionBase
    feed_forward: SwiGLUMLP
    seq_norm: nn.RMSNorm
    ffn_norm: nn.RMSNorm
    seq_post_norm: nn.RMSNorm
    ffn_post_norm: nn.RMSNorm

    def __init__(
        self,
        config: ModelConfig,
        *,
        key,
    ) -> None:
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.param_dtype)

        match config.seq_modeling_block:
            case "self_attention":
                seq_modeling_block_cls = Attention
            case "SWA":
                seq_modeling_block_cls = SlidingWindowAttention
            case _:
                raise NotImplementedError(f"Sequence Modeling Layer {config.seq_modeling_block} Not Implemented.")

        key_seq, key_ffn = jrandom.split(key, 2)
        self.seq_modeling_block = seq_modeling_block_cls(self.config, key=key_seq)
        self.feed_forward = SwiGLUMLP(self.config, key=key_ffn)
        self.seq_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)
        self.ffn_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)
        self.seq_post_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)
        self.ffn_post_norm = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)

    def seq_modeling_forward(
        self, seq_modeling_block_fn, rms_forward_fn, seq_norm, seq_modeling_block, seq_post_norm, hidden_states, state: nn.State, seq: Batch
    ):
        if self.config.pre_norm:
            seq_modeling_input = jax.vmap(lambda x: rms_forward_fn(seq_norm, x))(hidden_states)
        else:
            seq_modeling_input = hidden_states

        seq_modeling_hidden_states, state = seq_modeling_block_fn(seq_modeling_block, seq_modeling_input, seq, state)

        if self.config.post_norm:
            seq_modeling_hidden_states = jax.vmap(lambda x: rms_forward_fn(seq_post_norm, x))(seq_modeling_hidden_states)

        return seq_modeling_hidden_states, state

    def ffn_forward(self, feed_forward_fn, rms_forward_fn, ffn_norm, feed_forward, ffn_post_norm, hidden_states):
        if self.config.pre_norm:
            feed_forward_input = jax.vmap(lambda x: rms_forward_fn(ffn_norm, x))(hidden_states)
        else:
            feed_forward_input = hidden_states

        feed_forward_hidden_states = feed_forward_fn(feed_forward, feed_forward_input)

        if self.config.post_norm:
            feed_forward_hidden_states = jax.vmap(lambda x: rms_forward_fn(ffn_post_norm, x))(feed_forward_hidden_states)

        return feed_forward_hidden_states

    def __call__(self, hidden_states, state: nn.State, seq: Batch):
        config = self.config

        seq_modeling_block_fn = maybe_double_remat(
            self.seq_modeling_block.__class__.__call__,
            prevent_cse=True,
            policy_remat=config.remat_attention,
            policy_remat_bwd=config.remat_attention_bwd,
        )
        feed_forward_fn = maybe_double_remat(
            self.feed_forward.__class__.__call__, prevent_cse=True, policy_remat=config.remat_mlp, policy_remat_bwd=config.remat_mlp_bwd
        )
        rms_forward_fn = maybe_double_remat(nn.RMSNorm.__call__, prevent_cse=True, policy_remat=config.remat_rms, policy_remat_bwd=config.remat_rms_bwd)

        seq_modeling_output, state = self.seq_modeling_forward(
            seq_modeling_block_fn, rms_forward_fn, self.seq_norm, self.seq_modeling_block, self.seq_post_norm, hidden_states, state, seq
        )

        hidden_states = hidden_states + seq_modeling_output

        feed_forward_hidden_states = self.ffn_forward(feed_forward_fn, rms_forward_fn, self.ffn_norm, self.feed_forward, self.ffn_post_norm, hidden_states)

        hidden_states = hidden_states + feed_forward_hidden_states

        return hidden_states, state

    def weights(self):
        return eqx.filter(self, eqx.is_inexact_array)


class BlockCollection(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    blocks: Block  # vmap-ed init and application

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.config = config
        keys = jrandom.split(key, config.num_hidden_layers)
        self.blocks = jax.vmap(lambda k: Block(config, key=k))(keys)

    def __call__(
        self,
        hidden_states,
        state: nn.State,
        seq: Batch,
    ):
        substate = state.substate(self.blocks)
        block_fn = maybe_double_remat(
            self.blocks.__class__.__call__,
            prevent_cse=True,
            policy_remat=self.config.remat_block,
            policy_remat_bwd=self.config.remat_block_bwd,
        )

        def apply_block(x, block__substate):
            block, substate = block__substate
            x, substate = block_fn(block, x, substate, seq)
            return x, substate

        hidden_states, substate = scan_or_loop(apply_block, hidden_states, (self.blocks, substate), use_loop=self.config.unroll_block_scan)

        state = state.update(substate)

        outputs = BaseModelOutput(last_hidden_state=hidden_states, state=state)
        return outputs


class TransformerModel(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    wte: nn.Embedding
    dropout: nn.Dropout = eqx.field(static=True)
    ln_f: nn.RMSNorm
    h: BlockCollection

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.param_dtype)

        key_embed, key_block = jrandom.split(key, 2)

        vocab_size, embed_dim = config.vocab_size, config.hidden_size

        self.wte = nn.Embedding(
            weight=jax.nn.initializers.normal(stddev=self.config.initializer_range, dtype=self.param_dtype)(key_embed, (vocab_size, embed_dim)),
        )

        self.dropout = nn.Dropout(p=self.config.embd_pdrop)
        self.h = BlockCollection(self.config, key=key_block)
        self.ln_f = nn.RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)

    def __call__(
        self,
        state: nn.State,
        seq: Batch,
    ):
        rms_forward_fn = maybe_double_remat(
            nn.RMSNorm.__call__, prevent_cse=True, policy_remat=self.config.remat_rms, policy_remat_bwd=self.config.remat_rms_bwd
        )

        input_embeds = jax.vmap(self.wte)(seq.input_ids.astype(jnp.int32))
        input_embeds = input_embeds.astype(self.compute_dtype)
        hidden_states = self.dropout(input_embeds)
        outputs: BaseModelOutput = self.h(
            hidden_states,
            state=state,
            seq=seq,
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = jax.vmap(lambda x: rms_forward_fn(self.ln_f, x))(hidden_states)

        return BaseModelOutput(last_hidden_state=hidden_states, state=outputs.state)


class CausalLM(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    model: TransformerModel
    lm_head: NormalLinear | None

    class Output(eqx.Module):
        last_hidden_states: jnp.ndarray
        logits: jnp.ndarray
        new_state: nn.State

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.param_dtype)
        key_model, key_word_embeddings = jrandom.split(key, 2)

        self.model = TransformerModel(self.config, key=key_model)

        if not self.config.tie_word_embeddings:
            self.lm_head = NormalLinear(
                self.config,
                in_features=config.hidden_size,
                out_features=config.vocab_size,
                std=config.initializer_range,
                key=key_word_embeddings,
                name="lm_head",
            )
        else:
            self.lm_head = None

    def __call__(
        self,
        state: nn.State,
        seq: Batch,
    ) -> CausalLM.Output:
        outputs = self.model(state, seq)
        hidden_states = outputs.last_hidden_state
        assert hidden_states.dtype == self.compute_dtype, "The hidden_states before lm_head should be in compute_dtype"

        if self.config.tie_word_embeddings:
            shared_kernel = self.model.wte.weight.T
            hidden_states, shared_kernel = promote_dtype(hidden_states, shared_kernel, dtype=self.compute_dtype)
            lm_logits = hidden_states @ shared_kernel
        else:
            lm_logits = self.lm_head(hidden_states)

        return CausalLM.Output(last_hidden_states=hidden_states, logits=lm_logits, new_state=outputs.state)


class LanguageModel(eqx.Module):
    """Top-level model wrapper. Owns the CausalLM and provides loss/weight access."""

    class Output(eqx.Module):
        lm_output: CausalLM.Output
        state: nn.State

    class MetricType(StrEnum):
        loss = auto()
        token_nll_loss = auto()
        outer_grad_norm = auto()

    config: Config = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    step_index: nn.StateIndex
    language_model: CausalLM

    def __init__(
        self,
        config: Config,
        *,
        key: PRNGKeyArray,
    ):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.model.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.model.param_dtype)

        self.step_index = nn.StateIndex(jnp.array(0, dtype=jnp.int32))

        self.language_model = CausalLM(config.model, key=key)

    def lm_loss(
        self, seq: Batch, state: nn.State
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, nn.State]]:
        lm_outputs = self.language_model(state=state, seq=seq)

        loss, loss_pure_ce = cross_entropy_loss_and_accuracy(lm_outputs.logits, seq.target_tokens, seq.loss_masks)
        token_nll_loss = -token_log_probs(lm_outputs.logits, seq.target_tokens)

        return loss, (loss_pure_ce, token_nll_loss, lm_outputs.new_state)

    def loss_for_sequence(self, seq: Batch, state: nn.State) -> tuple[jnp.ndarray, dict[MetricType, jnp.ndarray]]:
        """Compute loss for a single sequence (no leading batch dimension)."""
        M = LanguageModel.MetricType
        metrics: dict[LanguageModel.MetricType, jnp.ndarray] = {}

        loss, (metrics[M.loss], metrics[M.token_nll_loss], _state) = self.lm_loss(seq, state)

        return loss, metrics

    def weights(self):
        """Get all weights of the model, including frozen parameters."""
        return eqx.filter(self, eqx.is_inexact_array)

    def trainable_parameters(self):
        """Get parameters that are trainable, excluding frozen parameters."""
        return filter_parameters(self.weights(), self.config.training.spec, "trainable parameters")

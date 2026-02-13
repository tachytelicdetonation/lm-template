from typing import override

import equinox as eqx
import jax
import jax.ad_checkpoint
import jax.numpy as jnp
import jax.random as jrandom
from equinox import nn
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray, PyTree

from lm.config import ModelConfig
from lm.model.data import Batch
from lm.utils.jax_utils import get_float_dtype_by_name, maybe_double_remat, promote_dtype, tree_rearrange


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype: jnp.dtype = jnp.float32) -> jnp.ndarray:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    t = jnp.arange(end)
    freqs = jnp.outer(t, freqs).astype(dtype)
    sin, cos = jnp.sin(freqs), jnp.cos(freqs)
    freqs_cis = jnp.complex64(cos + 1j * sin)
    return jnp.asarray(freqs_cis)


def apply_rotary_emb(x, freqs_cis: jnp.ndarray):
    input_dtype = x.dtype
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:-1], 1, *freqs_cis.shape[-1:]))
    reshape_x = x.astype(jnp.float32).reshape(*x.shape[:-1], -1, 2)
    x_ = jax.lax.complex(reshape_x[..., 0], reshape_x[..., 1])
    x_out = x_ * freqs_cis
    x_out = jnp.stack((jnp.real(x_out), jnp.imag(x_out)), axis=-1).reshape(*x_out.shape[:-1], -1)
    return x_out.astype(input_dtype)


class NormalLinear(eqx.Module):
    """Performs a linear transformation with truncated normal initialization."""

    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)

    weight: jax.Array
    name: str = eqx.field(static=True)

    def __init__(self, config: ModelConfig, in_features: int, out_features: int, *, name: str = "", std: float, key: PRNGKeyArray):
        self.compute_dtype = get_float_dtype_by_name(config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(config.param_dtype)
        self.in_features = in_features
        self.out_features = out_features

        self.weight = jrandom.normal(key, shape=(in_features, out_features), dtype=self.param_dtype) * std
        self.name = name

    @jax.named_scope("lm.transformer.NormalLinear")
    def __call__(self, x: Array) -> Array:
        if self.name:
            x = jax.ad_checkpoint.checkpoint_name(x, f"pre_promote_{self.name}")
        x, weight = promote_dtype(x, self.weight, dtype=self.compute_dtype)
        if self.name:
            x = jax.ad_checkpoint.checkpoint_name(x, f"pre_{self.name}")
        x = x @ weight
        if self.name:
            x = jax.ad_checkpoint.checkpoint_name(x, f"post_{self.name}")
        return x


class AttentionBase(eqx.Module):
    config: ModelConfig = eqx.field(static=True, repr=False)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    param_dtype: jnp.dtype = eqx.field(static=True)

    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    wq: NormalLinear
    wk: NormalLinear
    wv: NormalLinear
    wo: NormalLinear
    q_norm: nn.RMSNorm
    k_norm: nn.RMSNorm

    resid_dropout: nn.Dropout = eqx.field(static=True)

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        self.config = config
        self.compute_dtype = get_float_dtype_by_name(self.config.compute_dtype)
        self.param_dtype = get_float_dtype_by_name(self.config.param_dtype)

        embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = embed_dim // self.num_heads

        self.q_norm = nn.RMSNorm(self.head_dim, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=self.config.rms_norm_eps, use_bias=False, dtype=self.param_dtype)

        keys = jax.random.split(key, 4)

        self.wq, self.wk, self.wv, self.wo = (
            NormalLinear(
                self.config,
                in_features=embed_dim,
                out_features=embed_dim,
                std=config.initializer_range,
                key=w_key,
                name=name,
            )
            for w_key, name in zip(keys, ("wq", "wk", "wv", "wo"))
        )

        self.resid_dropout = nn.Dropout(p=config.resid_pdrop)

    @property
    def freqs_cis(self):
        with jax.ensure_compile_time_eval():
            freqs_cis = precompute_freqs_cis(
                self.head_dim,
                2 * self.config.seq_len,
                theta=self.config.rope_theta,
                dtype=jnp.float32,
            )

        return freqs_cis

    def _split_heads(self, x):
        return tree_rearrange(x, "... (head head_dim) -> ... head head_dim", head=self.num_heads, head_dim=self.head_dim)

    def _merge_heads(self, x):
        return tree_rearrange(x, "... head head_dim -> ... (head head_dim)", head=self.num_heads, head_dim=self.head_dim)

    def project_qkv(self, hidden_states):
        xq, xk, xv = self.wq(hidden_states), self.wk(hidden_states), self.wv(hidden_states)
        return xq, xk, xv

    def get_attention_input(self, hidden_states, position_ids):
        xq, xk, xv = self.project_qkv(hidden_states)  # [T,D]

        xq, xk, xv = self._split_heads((xq, xk, xv))  # [T,nh,d]

        if self.config.qk_norm:
            rms_forward_fn = maybe_double_remat(
                nn.RMSNorm.__call__, prevent_cse=True, policy_remat=self.config.remat_rms, policy_remat_bwd=self.config.remat_rms_bwd
            )
            xq = jax.vmap(jax.vmap(lambda x: rms_forward_fn(self.q_norm, x)))(xq)
            xk = jax.vmap(jax.vmap(lambda x: rms_forward_fn(self.k_norm, x)))(xk)

        xq, xk = self.apply_rope((xq, xk), position_ids=position_ids)
        return xq, xk, xv

    def apply_rope(self, xis: PyTree[jnp.ndarray], position_ids: jnp.ndarray) -> PyTree[jnp.ndarray]:
        freqs_cis = jnp.take(self.freqs_cis, position_ids, axis=0)
        apply_rotary_emb_fn = maybe_double_remat(
            apply_rotary_emb, prevent_cse=True, policy_remat=self.config.remat_rms, policy_remat_bwd=self.config.remat_rms_bwd
        )
        out_xis = jax.tree.map(lambda x: apply_rotary_emb_fn(x, freqs_cis), xis)
        return out_xis

    def get_attention_output(self, attn_output):
        o_output = self.wo(attn_output)
        attn_output = self.resid_dropout(o_output)
        return attn_output

    def __call__(self, *_args, **_kwargs) -> tuple[Array, nn.State]:
        raise NotImplementedError


class Attention(AttentionBase):
    """Full causal self-attention."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(config, key=key)

    @override
    def __call__(self, hidden_states, seq: Batch, state: nn.State):
        xq, xk, xv = self.get_attention_input(hidden_states, position_ids=jnp.arange(seq.shape[0]) if seq.position_ids is None else seq.position_ids)

        if self.config.force_flash:
            xq = jax.lax.with_sharding_constraint(xq, P(None, "state", None))
            xk = jax.lax.with_sharding_constraint(xk, P(None, "state", None))
            xv = jax.lax.with_sharding_constraint(xv, P(None, "state", None))

        attn_output = jax.nn.dot_product_attention(xq, xk, xv, is_causal=True, implementation="cudnn" if self.config.force_flash else None)

        if self.config.force_flash:
            attn_output = jax.lax.with_sharding_constraint(attn_output, P(None, "state", None))

        attn_output = self._merge_heads(attn_output)
        attn_output = self.get_attention_output(attn_output)

        return (attn_output, state)


class SlidingWindowAttention(AttentionBase):
    """Sliding window causal attention."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        key: PRNGKeyArray,
    ):
        super().__init__(config, key=key)

    @override
    def __call__(self, hidden_states, seq: Batch, state: nn.State):
        xq, xk, xv = self.get_attention_input(hidden_states, position_ids=jnp.arange(seq.shape[0]) if seq.position_ids is None else seq.position_ids)

        if self.config.force_flash:
            xq = jax.lax.with_sharding_constraint(xq, P(None, "state", None))
            xk = jax.lax.with_sharding_constraint(xk, P(None, "state", None))
            xv = jax.lax.with_sharding_constraint(xv, P(None, "state", None))

        attn_output = jax.nn.dot_product_attention(
            xq,
            xk,
            xv,
            local_window_size=(self.config.sliding_window_size - 1, 0),
            is_causal=True,
            implementation="cudnn" if self.config.force_flash else None,
        )

        if self.config.force_flash:
            attn_output = jax.lax.with_sharding_constraint(attn_output, P(None, "state", None))

        attn_output = self._merge_heads(attn_output)
        attn_output = self.get_attention_output(attn_output)

        return (attn_output, state)

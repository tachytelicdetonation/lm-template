from collections.abc import Callable
from typing import Any, TypeVar

import equinox as eqx
import jax
from jax.sharding import PartitionSpec as P
from jaxtyping import PyTree

from lm.config import Config

T = TypeVar("T", bound=PyTree)


def shard_fn[T: PyTree](tree: T, mesh: jax.sharding.Mesh, where_spec_pairs: list[tuple[Callable, P]]) -> T:
    for where, spec in where_spec_pairs:
        tree = eqx.tree_at(where, tree, replace_fn=lambda x: jax.lax.with_sharding_constraint(x, jax.NamedSharding(mesh, spec)), is_leaf=lambda x: x is None)
    return tree


class ModelSharding:
    def __init__(self, cfg: Config, mesh: jax.sharding.Mesh | None = None):
        self.config = cfg
        self.mesh = mesh

        if self.mesh is None:
            global_dev_num = jax.device_count()
            if cfg.training.n_data_parallel is None:
                assert global_dev_num % cfg.training.n_state_parallel == 0, "Number of devices must be divisible by state parallelism"
                n_data_parallel = global_dev_num // cfg.training.n_state_parallel
            else:
                n_data_parallel = cfg.training.n_data_parallel

            assert n_data_parallel * cfg.training.n_state_parallel == global_dev_num, (
                f"Data parallelism ({cfg.training.n_data_parallel}) and state parallelism ({cfg.training.n_state_parallel}) must match the number of devices ({global_dev_num})"
            )

            self.mesh = jax.make_mesh(axis_shapes=(n_data_parallel, cfg.training.n_state_parallel), axis_names=("data", "state"))

    def shard_params(self, model):
        shard_cfg = [
            (lambda m: (m.language_model.model.ln_f,), P("state")),
            (
                lambda m: (
                    m.language_model.model.wte,
                    m.language_model.model.h.blocks.seq_norm,
                    m.language_model.model.h.blocks.ffn_norm,
                    m.language_model.lm_head,
                ),
                P(None, "state"),
            ),
            (
                lambda m: (
                    m.language_model.model.h.blocks.seq_modeling_block.wq,
                    m.language_model.model.h.blocks.seq_modeling_block.wk,
                    m.language_model.model.h.blocks.seq_modeling_block.wv,
                    m.language_model.model.h.blocks.feed_forward.w1,
                    m.language_model.model.h.blocks.feed_forward.w3,
                ),
                P(None, "state", None),
            ),
            (
                lambda m: (
                    m.language_model.model.h.blocks.feed_forward.w2,
                    m.language_model.model.h.blocks.seq_modeling_block.wo,
                ),
                P(None, None, "state"),
            ),
        ]

        return shard_fn(model, self.mesh, shard_cfg)

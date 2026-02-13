import logging
from pathlib import Path

import equinox as eqx
import grain.python as grain
import jax
import jax.ad_checkpoint
import jax.numpy as jnp
import numpy as np
from equinox import nn
from optax import OptState
from tqdm import tqdm

from lm.config import Config
from lm.dataloader.lm_dataset import dummy_dataset, lm_dataset
from lm.infra.wandb_utils import WandbLogger
from lm.model.data import Batch
from lm.model.transformer import LanguageModel
from lm.optimizers import make_optimizer
from lm.utils.filter_utils import filter_apply_updates, get_filter_spec
from lm.utils.jax_utils import global_norm_safe, master_log, tree_rearrange, vmap_mean, welfords_online_mean

logger = logging.getLogger(__name__)

M = LanguageModel.MetricType


@eqx.filter_jit
@eqx.filter_vmap(axis_name="data_parallel", in_axes=(None, 0, None), out_axes=None)
def eval_step_fn(
    model: LanguageModel,
    seq: Batch,
    state: eqx.nn.State,
):
    """Get loss and token-wise NLL metrics for a single batch of data, reduced across all devices."""

    loss, metrics = model.loss_for_sequence(seq, state)

    _avg_loss, avg_metrics = jax.lax.pmean((loss, metrics), axis_name="data_parallel")

    return avg_metrics


class Evaluator:
    """Contains data loading and evaluation logic + state."""

    def __init__(
        self,
        global_batch_size: int,
        data_sharding: jax.sharding.NamedSharding,
        config: Config,
        wandb_logger: WandbLogger,
        log_dir: Path,
    ):
        self.train_holdout_loader = (
            lm_dataset(
                path=config.training.dataset_path,
                seq_len=config.training.seq_length,
                split=config.training.eval_split,
                global_batch_size=global_batch_size,
                seed=0,
                repeat=False,
                shuffle=False,
                bos_token_id=config.model.bos_token_id,
                eos_token_id=config.model.eos_token_id,
            )
            if not config.training.dummy_dataset
            else dummy_dataset(
                seq_len=config.training.seq_length,
                global_batch_size=global_batch_size,
                bos_token_id=config.model.bos_token_id,
                eos_token_id=config.model.eos_token_id,
                num_tokens=2**25,
            )
        )
        self.data_sharding = data_sharding
        self.config = config
        self.global_batch_size = global_batch_size
        self.wandb_logger = wandb_logger
        self.log_dir = log_dir

    def eval_fn(self, model: LanguageModel, state: eqx.nn.State, step: int):
        pid = jax.process_index()

        loader_dict = {"train_holdout": self.train_holdout_loader}

        def load_to_sharded_array(arr):
            return jax.make_array_from_process_local_data(sharding=self.data_sharding, local_data=arr, global_shape=(self.global_batch_size, *arr.shape[1:]))

        def eval_loader(path, ds: grain.MapDataset):
            batch_loader = ds.to_iter_dataset().map(lambda batch: jax.tree.map(load_to_sharded_array, batch))
            loader_key = path[0].key

            results = []
            for batch in tqdm(batch_loader, desc=f"Evaluating on sequence {loader_key}", total=len(ds), disable=pid != 0):
                result = eval_step_fn(model, batch, state)
                results.append(result)

            results_mean = jax.tree.map(lambda *x: np.asarray(jnp.mean(jnp.stack(x), axis=0, dtype=jnp.float32)), *results)

            return results_mean

        eval_metrics = jax.tree.map_with_path(eval_loader, loader_dict)

        self.log_eval_results(eval_metrics, step)

    def log_eval_results(self, eval_metrics, step):
        for eval_name, v in eval_metrics.items():
            for metric_name, metric in v.items():
                if metric_name == M.loss:
                    master_log(logger, f"Eval -- {eval_name}/{metric_name}: {metric.mean()}")
                    self.wandb_logger.log({f"{eval_name}/{metric_name}": metric}, step)

                else:
                    save_dir = self.log_dir / f"{eval_name}_{metric_name}.npy"
                    if metric_name == M.token_nll_loss:
                        self.wandb_logger.log_token_nll_loss(metric, step, eval_name)
                    np.save(save_dir, metric)
                    self.wandb_logger.save(save_dir, self.log_dir)


@eqx.filter_jit(donate="all-except-first")
@eqx.filter_vmap(in_axes=(None, None, None, 0, None), out_axes=None, axis_name="data_parallel")
def train_on_sequence(
    state: nn.State, model: LanguageModel, opt_state: OptState, batch: Batch, cfg: Config
) -> tuple[LanguageModel, OptState, jnp.ndarray, dict[LanguageModel.MetricType, jnp.ndarray]]:
    """Train the model for a single step on a sequence."""
    M = LanguageModel.MetricType
    assert batch.shape[0] % cfg.training.accum_steps == 0, (
        f"Gradient accumulation steps should divide the per-device batch size, got {batch.shape[0]=} !% {cfg.training.accum_steps=}"
    )

    batch = tree_rearrange(batch, "(accum batch) ... -> accum batch ...", accum=cfg.training.accum_steps)

    loss_fn = lambda model, b: vmap_mean(lambda seq: LanguageModel.loss_for_sequence(model, seq, state), b, axis_name="batch")

    meta_grad_fn = lambda batch: eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, batch)

    meta_grad_fn = lambda batch, fun=meta_grad_fn: welfords_online_mean(fun, batch)
    (loss, metrics), grads = meta_grad_fn(batch)

    avg_loss, avg_metrics, avg_grads = jax.lax.pmean((loss, metrics, grads), axis_name="data_parallel")

    avg_grads = avg_grads.trainable_parameters()
    avg_outer_gnorm = global_norm_safe(avg_grads)
    avg_metrics[M.outer_grad_norm] = avg_outer_gnorm

    tx, _ = make_optimizer(cfg.training.optimizer)
    updates, opt_state = tx.update(avg_grads, opt_state, model.trainable_parameters())

    model = filter_apply_updates(model, updates)

    return (model, opt_state, avg_loss, avg_metrics)

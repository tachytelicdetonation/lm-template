import dataclasses
import json
from pathlib import Path
from typing import Any, TypeVar

import equinox as eqx
import grain.python as grain
import jax
import orbax.checkpoint as ocp
from etils import epath
from grain._src.python import data_loader
from grain._src.python.dataset import dataset
from optax import OptState
from orbax.checkpoint import options as ocp_opt

from lm.config import Config, TrainingConfig

IteratorType = TypeVar("IteratorType", data_loader.DataLoaderIterator, dataset.DatasetIterator)


class CustomPyGrainCheckpointHandler(grain.PyGrainCheckpointHandler):
    """Orbax CheckpointHandler for PyGrain iterators."""

    def save(
        self,
        directory: epath.Path,
        args: Any = None,
    ):
        """Saves the given iterator to the checkpoint in `directory`."""
        item = args.item
        if isinstance(item, dataset.DatasetIterator):
            state = json.dumps(item.get_state(), indent=4)
        else:
            state = item.get_state().decode()
        filename = directory / "global_batch_progress.json"

        if jax.process_index() == 0:
            filename.write_text(state)

    def restore(
        self,
        directory: epath.Path,
        args: Any = None,
    ) -> IteratorType:
        """Restores the given iterator from the checkpoint in `directory`."""
        item = args.item
        filename = directory / "global_batch_progress.json"
        if not filename.exists():
            raise ValueError(f"File {filename} does not exist.")
        state = filename.read_text()
        if isinstance(item, dataset.DatasetIterator):
            state = json.loads(state)
        else:
            state = state.encode()
        item.set_state(state)
        return item


@ocp.args.register_with_handler(CustomPyGrainCheckpointHandler, for_save=True)
@dataclasses.dataclass
class CustomPyGrainCheckpointSave(ocp.args.CheckpointArgs):
    item: Any


@ocp.args.register_with_handler(CustomPyGrainCheckpointHandler, for_restore=True)
@dataclasses.dataclass
class CustomPyGrainCheckpointRestore(ocp.args.CheckpointArgs):
    item: Any


class Checkpointer:
    def __init__(self, config: Config, for_saving: bool = True):
        self.config = config

        if for_saving:
            checkpoint_path = config.checkpoint.checkpoint_dir
        else:
            checkpoint_path = config.checkpoint.resume_checkpoint_dir

        if not checkpoint_path.startswith("gs://"):
            checkpoint_path = Path(checkpoint_path).resolve()

        handler_registry = ocp.DefaultCheckpointHandlerRegistry()
        handler_registry.add("train_ds_iter", CustomPyGrainCheckpointRestore, CustomPyGrainCheckpointHandler)
        handler_registry.add("train_ds_iter", CustomPyGrainCheckpointSave, CustomPyGrainCheckpointHandler)
        handler_registry.add("opt_state", ocp.args.StandardRestore, ocp.StandardCheckpointHandler)
        handler_registry.add("opt_state", ocp.args.StandardSave, ocp.StandardCheckpointHandler)
        handler_registry.add("model_weights", ocp.args.StandardRestore, ocp.StandardCheckpointHandler)
        handler_registry.add("model_weights", ocp.args.StandardSave, ocp.StandardCheckpointHandler)

        mp_opts = ocp_opt.MultiprocessingOptions(primary_host=0)
        ckpt_opts = ocp.CheckpointManagerOptions(multiprocessing_options=mp_opts)

        self.manager = ocp.CheckpointManager(
            checkpoint_path,
            options=ckpt_opts,
            handler_registry=handler_registry,
        )

    def save_checkpoint(self, step: int, model: eqx.Module, opt_state: OptState, train_ds_iter, is_milestone: bool = False):
        model_weights = model.weights()

        self.manager.save(
            step=step,
            args=ocp.args.Composite(
                opt_state=ocp.args.StandardSave(opt_state),
                model_weights=ocp.args.StandardSave(model_weights),
                train_ds_iter=CustomPyGrainCheckpointSave(train_ds_iter),
            ),
            force=is_milestone,
        )

    def checkpoint_exists(self) -> bool:
        return self.manager.latest_step() is not None

    def load_checkpoint(self, targets, restore: TrainingConfig.LoadPart, step=None):
        if step is None:
            step = self.manager.latest_step()

        if step is None:
            raise FileNotFoundError(f"No checkpoints found at {self.manager.directory}")

        model_weights_metadata = self.manager.item_metadata(step)["model_weights"]
        model_weights_target = fetch_from_eqx_module(model_weights_metadata, targets["model_weights"])[0]

        opt_state_metadata = self.manager.item_metadata(step)["opt_state"]
        opt_state_target = fetch_from_eqx_module(opt_state_metadata, targets["opt_state"])[0]

        if restore == TrainingConfig.LoadPart.all:
            restored = self.manager.restore(
                step=step,
                args=ocp.args.Composite(
                    opt_state=ocp.args.StandardRestore(opt_state_target),
                    model_weights=ocp.args.StandardRestore(model_weights_target),
                    train_ds_iter=CustomPyGrainCheckpointRestore(targets["train_ds_iter"]),
                ),
            )
            return {
                "opt_state": restored["opt_state"],
                "model_weights": restored["model_weights"],
                "train_ds_iter": restored["train_ds_iter"],
            }
        elif restore == TrainingConfig.LoadPart.params:
            restored = self.manager.restore(
                step=step,
                args=ocp.args.Composite(
                    model_weights=ocp.args.StandardRestore(model_weights_target),
                ),
            )
            return {"model_weights": restored["model_weights"]}
        else:
            raise ValueError(f"Invalid restore option: {restore:r}")

    def wait_until_finished(self):
        self.manager.wait_until_finished()

    def close(self):
        self.manager.close()


M = TypeVar("M", bound=eqx.Module)


def unify_dict_with_eqx_module[M: eqx.Module](d: dict, module: M) -> tuple[M, list[str]]:
    """
    Create an Equinox module from the data in dictionary `d`, relying on the structure being the same.
    Values missing in the dictionary will be taken from the module.
    """
    from jax._src.lib import pytree

    weights_map = {p: v for p, v in jax.tree.flatten_with_path(d)[0]}

    not_found_paths = []

    def find_weight(path, value):
        dict_path = tuple(pytree.DictKey(p.name) if isinstance(p, pytree.GetAttrKey) else p for p in path)
        if dict_path in weights_map:
            return weights_map[dict_path]
        else:
            not_found_paths.append(jax.tree_util.keystr(path))
            return value

    new_module = jax.tree.map_with_path(find_weight, module)

    if not_found_paths:
        import warnings

        warnings.warn(f"Could not find the following paths in the dictionary: {not_found_paths}")

    return new_module, not_found_paths


def fetch_from_eqx_module[M: eqx.Module](d: dict, module: M) -> tuple[M, list[str]]:
    """Fetch values from the module and put them in the dictionary `d`."""
    from jax._src.lib import pytree

    eqx_map = {p: v for p, v in jax.tree.flatten_with_path(module)[0]}

    not_found_paths = []

    def find_weight(path, value):
        dict_path = tuple(pytree.GetAttrKey(p.key) if isinstance(p, pytree.DictKey) else p for p in path)
        if dict_path in eqx_map:
            new_value = eqx_map[dict_path]
            assert new_value.shape == value.shape, f"Shape mismatch for {jax.tree_util.keystr(path)}: {new_value.shape} != {value.shape}"
            return new_value
        else:
            not_found_paths.append(jax.tree_util.keystr(path))
            return value

    new_dict = jax.tree.map_with_path(find_weight, d)
    if not_found_paths:
        import warnings

        warnings.warn(f"Could not find the following paths in the checkpoint module: {not_found_paths}")
    return new_dict, not_found_paths

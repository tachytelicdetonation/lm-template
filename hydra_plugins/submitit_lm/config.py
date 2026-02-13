# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class BaseQueueConf:
    """Configuration shared by all executors"""

    submitit_folder: str = "${hydra.sweep.dir}/.submitit/%j"

    # maximum time for the job in minutes
    timeout_min: int = 60
    # number of cpus to use for each task
    cpus_per_task: int | None = None
    # number of gpus to use on each node
    gpus_per_node: int | None = None
    # number of tasks to spawn on each node
    tasks_per_node: int = 1
    # memory to reserve for the job on each node (in GB)
    mem_gb: int | None = None
    # number of nodes to use for the job
    nodes: int = 1
    # name of the job
    name: str = "${hydra.job.name}"
    # redirect stderr to stdout
    stderr_to_stdout: bool = False
    # How to invoke Python on the target machine (e.g., "uv run python")
    slurm_python: str | None = None


@dataclass
class SlurmQueueConf(BaseQueueConf):
    """Slurm configuration overrides and specific parameters"""

    _target_: str = "hydra_plugins.submitit_lm.submitit_launcher.SlurmLauncher"

    # Params are used to configure sbatch, for more info check:
    # https://github.com/facebookincubator/submitit/blob/master/submitit/slurm/slurm.py

    # Following parameters are slurm specific
    # More information: https://slurm.schedmd.com/sbatch.html
    #
    # slurm partition to use on the cluster
    partition: str | None = None
    qos: str | None = None
    comment: str | None = None
    constraint: str | None = None
    exclude: str | None = None
    gres: str | None = None
    cpus_per_gpu: int | None = None
    gpus_per_task: int | None = None
    mem_per_gpu: str | None = None
    mem_per_cpu: str | None = None
    account: str | None = None

    # Following parameters are submitit specifics
    #
    # USR1 signal delay before timeout
    signal_delay_s: int = 120
    # Maximum number of retries on job timeout.
    # Change this only after you confirmed your code can handle re-submission
    # by properly resuming from the latest stored checkpoint.
    # check the following for more info on slurm_max_num_timeout
    # https://github.com/facebookincubator/submitit/blob/master/docs/checkpointing.md
    max_num_timeout: int = 0
    # Useful to add parameters which are not currently available in the plugin.
    # Eg: {"mail-user": "blublu@fb.com", "mail-type": "BEGIN"}
    additional_parameters: dict[str, Any] = field(default_factory=dict)
    # Maximum number of jobs running in parallel
    array_parallelism: int = 256
    # A list of commands to run in sbatch befure running srun
    setup: list[str] | None = None
    srun_args: list[str] | None = None


@dataclass
class LocalQueueConf(BaseQueueConf):
    _target_: str = "hydra_plugins.submitit_lm.submitit_launcher.LocalLauncher"


# finally, register two different choices:
ConfigStore.instance().store(
    group="hydra/launcher",
    name="submitit_local_lm",
    node=LocalQueueConf(),
    provider="submitit_launcher_lm",
)


ConfigStore.instance().store(
    group="hydra/launcher",
    name="submitit_slurm_lm",
    node=SlurmQueueConf(),
    provider="submitit_launcher_lm",
)

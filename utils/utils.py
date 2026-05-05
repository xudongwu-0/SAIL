import wandb
from dataclasses import asdict
import numpy as np
from utils import CONFIGS

TASK_CONFIGS = CONFIGS.tasks
WANDB_CONFIGS = CONFIGS.services.wandb
PARAMS_CONFIGS = CONFIGS.cdpo.params


# Format the command line arguments
def format_args(value):
    if isinstance(value, str):
        return value
    elif isinstance(value, bool):
        return "yes" if value else "no"
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        if (value * 100).is_integer():
            return "{:.2f}".format(value)
        else:
            return "{:.2e}".format(value)
    return value


# Format the run name
def format_run_name(pipeline, model, dataset, extra_params):
    if pipeline == "SFT":
        configs = ""
    else:
        if pipeline not in PARAMS_CONFIGS:
            raise ValueError(f"Unknown pipeline name: {pipeline}")
        required_params = PARAMS_CONFIGS[pipeline]
        configs = "".join(
            [
                param_name
                + (
                    f"{int(extra_params[param_name])}"
                    if isinstance(extra_params[param_name], int)
                    and not isinstance(extra_params[param_name], bool)
                    else f"{extra_params[param_name]:.2f}"
                )
                for param_name in required_params
            ]
        )
    return pipeline + "_" + model + "_" + dataset + ("_" if configs else "") + configs


# Generate sweep tasks
def generate_sweep_tasks():
    dpo_tasks = []
    tag = TASK_CONFIGS["tag"]
    models = TASK_CONFIGS["model"]
    datasets = TASK_CONFIGS["dataset"]
    for task in TASK_CONFIGS["tasks"]:
        pipeline = task["pipeline"]
        extra_fields_list = [{}]
        for param, values in task.items():
            if param != "pipeline":
                new_extra_fields_list = []
                for extra_fields in extra_fields_list:
                    for value in values:
                        new_extra_fields = extra_fields.copy()
                        new_extra_fields[param] = value
                        new_extra_fields_list.append(new_extra_fields)
                extra_fields_list = new_extra_fields_list
        for model in models:
            for dataset in datasets:
                for extra_fields in extra_fields_list:
                    task_config = {
                        "pipeline": pipeline,
                        "model": model,
                        "dataset": dataset,
                        "tag": tag,
                        **extra_fields,
                    }
                    dpo_tasks.append(task_config)
    sft_tasks = [
        dict(t)
        for t in {
            tuple(d.items())
            for d in [
                {k: v for k, v in t.items() if k in ["model", "dataset", "tag"]}
                for t in dpo_tasks
            ]
        }
    ]
    commands = []
    if "SFT" in TASK_CONFIGS["pipelines"]:
        for task in sft_tasks:
            commands.append(
                "cdpo sft "
                + " ".join([f"--{k} {format_args(v)}" for k, v in task.items()]),
            )
    for task in dpo_tasks:
        for pipeline in [
            p
            for p in ["DPO", "GEN", "EVALREWARD", "EVALGPT"]
            if p in TASK_CONFIGS["pipelines"]
        ]:
            commands.append(
                f"cdpo {pipeline.lower()} "
                + " ".join([f"--{k} {format_args(v)}" for k, v in task.items()]),
            )
    return commands


# Initialize wandb
def wandb_init(run_name, script_args, training_args):
    wandb.init(
        project=WANDB_CONFIGS["project"],
        entity=WANDB_CONFIGS["team"],
        name=script_args.tag + "-" + run_name,
        config={
            k: v
            for args in [
                script_args,
                training_args,
            ]
            for k, v in asdict(args).items()
        },
    )


# Sample every k samples from the dataset with batched sampling
def sample_every_k_batched(dataset, every_k, batch_size):
    # Ensure every_k is a positive float
    every_k = float(every_k)
    assert every_k > 0, "every_k must be a positive value"
    num_rows = len(dataset)

    if every_k >= 1:
        # Round every_k to the nearest integer
        step = int(np.round(every_k))
        total_samples = len(range(0, num_rows, step))
    else:
        # Calculate round(1 / every_k) and sample each row for that many times
        step = int(np.round(1 / every_k))
        total_samples = num_rows * step

    # Calculate the total number of iterations, considering the batch size
    num_iters = (total_samples + batch_size - 1) // batch_size

    def generator():
        if every_k >= 1:
            sampled_indices = list(range(0, num_rows, step))
        else:
            sampled_indices = list(np.repeat(np.arange(num_rows), step))

        # Yield batches
        for i in range(0, len(sampled_indices), batch_size):
            batch_indices = sampled_indices[i : i + batch_size]
            batch_samples = [
                dataset[int(idx)] if idx < num_rows else None for idx in batch_indices
            ]  # Handles out-of-bound indices and ensures proper key type
            # Filter out any None values that may appear if indices go out of bounds
            valid_batch_indices_samples = [
                (idx, samp)
                for idx, samp in zip(batch_indices, batch_samples)
                if samp is not None
            ]
            if valid_batch_indices_samples:
                yield zip(
                    *valid_batch_indices_samples
                )  # Unzip to separate indices and samples

    return generator(), num_iters

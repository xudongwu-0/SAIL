import os
import sys
import click
import subprocess

from utils import CONFIGS, format_args, format_run_name, generate_sweep_tasks

DEVICE_CONFIGS = CONFIGS.devices.devices
OPTIM_CONFIGS = CONFIGS.pipelines.optim
PARAMS_CONFIGS = CONFIGS.cdpo.params
CACHE_CONFIGS = CONFIGS.utils.cache
TASK_CONFIGS = CONFIGS.tasks


@click.group()
def cli():
    """Correct-DPO Package"""
    pass


def parse_extra_args(pipeline, extra_params):
    required_params = PARAMS_CONFIGS[pipeline]
    parsed_extra_params = {}
    i = 0
    while i < len(extra_params):
        arg = extra_params[i]
        if arg.startswith("--"):
            param_name = arg[2:]
            if param_name in required_params:
                if i + 1 < len(extra_params):
                    param_value = extra_params[i + 1]
                    if param_value.lower() in ["yes", "true"]:
                        parsed_extra_params[param_name] = True
                    elif param_value.lower() in ["no", "false"]:
                        parsed_extra_params[param_name] = False
                    else:
                        try:
                            parsed_extra_params[param_name] = int(param_value)
                        except ValueError:
                            try:
                                parsed_extra_params[param_name] = float(param_value)
                            except ValueError:
                                parsed_extra_params[param_name] = param_value
                    i += 1
                else:
                    raise click.BadParameter(
                        f"Missing value for parameter '{param_name}'"
                    )
            else:
                raise click.BadParameter(
                    f"Unexpected parameter for pipeline '{pipeline}': {param_name}"
                )
        else:
            raise click.BadParameter(f"Invalid argument: {arg}")
        i += 1

    missing_params = set(required_params) - set(parsed_extra_params.keys())
    if missing_params:
        raise click.BadParameter(
            f"Missing required parameters for pipeline '{pipeline}': {', '.join(missing_params)}"
        )
    return parsed_extra_params


def get_accelerate_params(pipeline, gres):
    return {
        "config_file": DEVICE_CONFIGS["config_file"][pipeline],
        "num_processes": int(gres.split(":")[1]),
    }


def get_optimizer_params(pipeline, model, dataset):
    # Get the default values for the pipeline
    defaults = OPTIM_CONFIGS["defaults"]["DPO" if pipeline != "SFT" else "SFT"]
    num_train_epochs = int(defaults["num_train_epochs"])
    learning_rate = float(defaults["learning_rate"])

    # Apply multipliers based on the dataset and model
    if dataset in OPTIM_CONFIGS["multipliers"]["num_train_epochs"]:
        num_train_epochs *= OPTIM_CONFIGS["multipliers"]["num_train_epochs"][dataset]

    if model in OPTIM_CONFIGS["multipliers"]["learning_rate"]:
        learning_rate *= OPTIM_CONFIGS["multipliers"]["learning_rate"][model]

    return {"num_train_epochs": int(num_train_epochs), "learning_rate": learning_rate}


def get_batch_size_params(pipeline, gres):
    try:
        return DEVICE_CONFIGS["pipelines"][pipeline][gres.split(":")[0]]
    except:
        KeyError(f"GPU device {gres} not found in the configuration file.")


def call(script_name, accelerate_kwargs={}, **scipt_kwargs):
    env = os.environ.copy()
    if accelerate_kwargs:
        executable = ["accelerate", "launch"]
        env["OMP_NUM_THREADS"] = str(accelerate_kwargs["num_processes"])
    else:
        executable = [sys.executable]
    scipt_kwargs = {
        key: format_args(value)
        for key, value in scipt_kwargs.items()
        if value is not None
    }
    subprocess.run(
        [
            *executable,
            *[f"--{key}={value}" for key, value in accelerate_kwargs.items()],
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"pipelines/{script_name}.py",
            ),
            *[f"--{key}={value}" for key, value in scipt_kwargs.items()],
        ],
        env=env,
        check=True,
    )


@click.command()
@click.option(
    "-t", "--tag", required=True, help="Tag for the experiment to be deleted."
)
def clear(tag):
    """CLEAR: Delete experiment repos by tag."""
    call("clear", tag=tag)


@click.command()
@click.option(
    "-p", "--prefix", default=None, help="Prefix for the experiment to be preprocessed."
)
def prep(prefix):
    """PREP: Preprocess datasets and upload."""
    call("clear", clear_datasets=True)
    call("preprocess", prefix=prefix)


@click.command()
@click.option("-m", "--model", required=True, help="Model name.")
@click.option("-d", "--dataset", required=True, help="Dataset name.")
@click.option("-t", "--tag", required=True, help="Tag for the experiment.")
@click.option("-g", "--gres", default=DEVICE_CONFIGS["local"], help="GPU resources.")
def sft(model, dataset, tag, gres):
    """SFT: Supervised Fine-tuning."""
    accelerate_kwargs = get_accelerate_params("SFT", gres)
    script_kwargs = {
        "model": model,
        "dataset": dataset,
        "tag": tag,
    }
    script_kwargs |= get_optimizer_params("SFT", model, dataset)
    script_kwargs |= get_batch_size_params("SFT", gres)
    call("sft", accelerate_kwargs, **script_kwargs)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("-p", "--pipeline", required=True, help="Pipeline name.")
@click.option("-m", "--model", required=True, help="Model name.")
@click.option("-d", "--dataset", required=True, help="Dataset name.")
@click.option("-t", "--tag", required=True, help="Tag for the experiment.")
@click.option("-g", "--gres", default=DEVICE_CONFIGS["local"], help="GPU resources.")
@click.argument("extra_params", nargs=-1, type=click.UNPROCESSED)
def dpo(pipeline, model, dataset, tag, gres, extra_params):
    """DPO: all variants of Direct Preference Optimization."""
    extra_params = parse_extra_args(pipeline, extra_params)
    accelerate_kwargs = get_accelerate_params("DPO", gres)
    script_kwargs = {
        "pipeline": pipeline,
        "model": model,
        "dataset": dataset,
        "tag": tag,
    }
    script_kwargs |= extra_params
    script_kwargs |= get_optimizer_params(pipeline, model, dataset)
    script_kwargs |= get_batch_size_params("DPO", gres)
    call("dpo", accelerate_kwargs, **script_kwargs)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("-p", "--pipeline", required=True, help="Pipeline name.")
@click.option("-m", "--model", required=True, help="Model name.")
@click.option("-d", "--dataset", required=True, help="Dataset name.")
@click.option("-t", "--tag", required=True, help="Tag for the experiment.")
@click.option("-g", "--gres", default=DEVICE_CONFIGS["local"], help="GPU resources.")
@click.argument("extra_params", nargs=-1, type=click.UNPROCESSED)
def gen(pipeline, model, dataset, tag, gres, extra_params):
    """GEN: generate responses."""
    extra_params = parse_extra_args(pipeline, extra_params)
    run = format_run_name(pipeline, model, dataset, extra_params)
    accelerate_kwargs = get_accelerate_params("GEN", gres)
    script_kwargs = {
        "run": run,
        "tag": tag,
    }
    script_kwargs |= get_batch_size_params("GEN", gres)
    call("generate", accelerate_kwargs, **script_kwargs)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("-p", "--pipeline", required=True, help="Pipeline name.")
@click.option("-m", "--model", required=True, help="Model name.")
@click.option("-d", "--dataset", required=True, help="Dataset name.")
@click.option("-t", "--tag", required=True, help="Tag for the experiment.")
@click.option("-g", "--gres", default=DEVICE_CONFIGS["local"], help="GPU resources.")
@click.argument("extra_params", nargs=-1, type=click.UNPROCESSED)
def evalreward(pipeline, model, dataset, tag, gres, extra_params):
    """EVALREWARD: evalute responses with reference reward model."""
    extra_params = parse_extra_args(pipeline, extra_params)
    run = format_run_name(pipeline, model, dataset, extra_params)
    accelerate_kwargs = get_accelerate_params("GEN", gres)
    script_kwargs = {
        "run": run,
        "tag": tag,
    }
    script_kwargs |= get_batch_size_params("EVALREWARD", gres)
    call("evalreward", accelerate_kwargs, **script_kwargs)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option("-p", "--pipeline", required=True, help="Pipeline name.")
@click.option("-m", "--model", required=True, help="Model name.")
@click.option("-d", "--dataset", required=True, help="Dataset name.")
@click.option("-t", "--tag", required=True, help="Tag for the experiment.")
@click.option("-g", "--gres", default=DEVICE_CONFIGS["local"], help="GPU resources.")
@click.argument("extra_params", nargs=-1, type=click.UNPROCESSED)
def evalgpt(pipeline, model, dataset, tag, gres, extra_params):
    """EVALGPT: evalute responses with GPT."""
    extra_params = parse_extra_args(pipeline, extra_params)
    run = format_run_name(pipeline, model, dataset, extra_params)
    script_kwargs = {
        "run": run,
        "tag": tag,
    }
    call("evalgpt", **script_kwargs)


# @click.command()
# @click.option("-i", "--index", default=None, help="Job index.")
# def execute(index):
#     index = "" if index is None else index
#     commands = generate_sweep_tasks()
#     with open(
#         os.path.join(
#             CACHE_CONFIGS["task_cache_dir"],
#             f"local{index}.sh",
#         ),
#         "w",
#     ) as f:
#         f.write("#!/bin/bash\n")
#         f.write("# source ./venv/bin/activate\n") #注释掉原来的要求激活环境
#         for command in commands:
#             f.write(command + "\n")
#     subprocess.run(
#         ["bash", os.path.join(CACHE_CONFIGS["task_cache_dir"], f"local{index}.sh")],
#         check=True,
#     )


@click.command()
@click.option("-i", "--index", default=None, help="Job index.")
@click.option("--out", "out_path", default=None, help="Where to write the bash script.")
@click.option("--no-run", is_flag=True, help="Only generate the script; do not execute.")
def execute(index, out_path, no_run):
    index = "" if index is None else str(index)
    commands = generate_sweep_tasks()

    default_path = os.path.join(CACHE_CONFIGS["task_cache_dir"], f"local{index}.sh")
    script_path = out_path if out_path is not None else default_path

    os.makedirs(os.path.dirname(script_path), exist_ok=True)

    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -euo pipefail\n")
        f.write("# source ./venv/bin/activate\n")
        for command in commands:
            f.write(command + "\n")

    # make it executable
    subprocess.run(["chmod", "+x", script_path], check=True)

    print(f"[cdpo] Script generated: {script_path}")
    print(f"[cdpo] Num commands: {len(commands)}")
    if no_run:
        print("[cdpo] Not executing (---no-run). Review the script then run it manually.")
        return

    subprocess.run(["bash", script_path], check=True)


# Task commands
cli.add_command(execute)
# Pipeline commands
cli.add_command(sft)
cli.add_command(dpo)
cli.add_command(gen)
cli.add_command(evalreward)
cli.add_command(evalgpt)
# Utility commands
cli.add_command(clear)
cli.add_command(prep)
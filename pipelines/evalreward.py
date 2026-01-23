# pipelines/evalreward.py
import os
import sys
from dataclasses import dataclass, field
from tqdm.auto import tqdm
from huggingface_hub import HfApi
import torch
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
)
from transformers import logging
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset, DatasetDict


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CONFIGS, sample_every_k_batched
from safe_rlhf.models import AutoModelForScore

HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
HFAPI = HfApi(HUGGINGFACE_CONFIGS["token"])
REWARD_CONFIGS = CONFIGS.evaluations.reward
CACHE_CONFIGS = CONFIGS.utils.cache

accelerator = Accelerator()
tqdm.pandas()
logging.set_verbosity_error()


@dataclass
class ScriptArguments:
    run_name: str = field(
        metadata={"help": "run name to evaluate"},
    )
    tag: str = field(
        metadata={"help": "tag for the experiment"},
    )
    every_k: int = field(
        default=1,
        metadata={
            "help": "evaluate every k samples, if a fraction, evaluate each sample 1/k times"
        },
    )
    per_device_evalreward_batch_size: int = field(
        default=16,
        metadata={"help": "batch size per device"},
    )
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "maximum sequence length, sequences will be right padded (and possibly truncated)"
        },
    )
    padding: bool = field(
        default=True,
        metadata={"help": "use padding for tokenizer"},
    )
    truncation: bool = field(
        default=True,
        metadata={"help": "allow truncation for tokenizer"},
    )
    score_model_id: str = field(
        default="none",
        metadata={"help": "model id for scoring, do not change"},
    )
    model_cache_dir: str = field(
        default=CACHE_CONFIGS["model_cache_dir"],
        metadata={"help": "model cache directory"},
    )
    dataset_cache_dir: str = field(
        default=CACHE_CONFIGS["dataset_cache_dir"],
        metadata={"help": "dataset cache directory"},
    )


# Load dataset and remove unnecessary columns
def load_generated_dataset(script_args):
    response_dataset = load_dataset(
        HUGGINGFACE_CONFIGS["prefix"]["evaluations"] + script_args.run_name,
        name=script_args.tag,
        cache_dir=script_args.dataset_cache_dir,
    )
    return response_dataset["default"]


# Load model and tokenizer for inference
def load_score_model(script_args):
    if script_args.score_model_id.startswith("PKU-Alignment"):
        model = AutoModelForScore.from_pretrained(
            script_args.score_model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": accelerator.local_process_index},
            # Update transformers package to >=4.38.0 and no need to use trust_remote_code
            trust_remote_code=False,
            use_cache=True,
            cache_dir=script_args.model_cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.score_model_id,
            model_max_length=script_args.model_max_length,
            padding_side="left",
            use_fast=True,
            cache_dir=script_args.model_cache_dir,
        )
    elif script_args.score_model_id.startswith("openbmb"):
        model = AutoModel.from_pretrained(
            script_args.score_model_id,
            torch_dtype=torch.bfloat16,
            device_map={"": accelerator.local_process_index},
            # This is needed as EurusRewardModel is not in transformers' model registry
            trust_remote_code=True,
            use_cache=True,
            cache_dir=script_args.model_cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.score_model_id,
            model_max_length=script_args.model_max_length,
            padding_side="left",
            use_fast=True,
            cache_dir=script_args.model_cache_dir,
        )
    else:
        raise ValueError("No reward model found for the dataset")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# Evaluate reward of responses
def evaluate_reward(model, tokenizer, response_dataset, script_args):
    generator, num_iters = sample_every_k_batched(
        response_dataset,
        script_args.every_k,
        batch_size=script_args.per_device_evalreward_batch_size,
    )
    # Progress bar only on main process
    pbar = tqdm(total=num_iters, disable=not accelerator.is_local_main_process)
    all_indices_and_samples = list(generator)
    with accelerator.split_between_processes(
        all_indices_and_samples
    ) as process_indices_and_samples:
        # Inference in batch
        result = {}
        for indices, samples in process_indices_and_samples:
            with torch.no_grad():
                inputs = tokenizer(
                    [sample["prompt"] + sample["response"] for sample in samples],
                    max_length=script_args.model_max_length,
                    truncation=script_args.truncation,
                    padding=script_args.padding,
                    return_tensors="pt",
                ).to(model.device)
                outputs = model(**inputs)
                if script_args.score_model_id.startswith("PKU-Alignment"):
                    scores = outputs.end_scores.cpu().flatten().tolist()
                elif script_args.score_model_id.startswith("openbmb"):
                    scores = outputs.cpu().flatten().tolist()
                else:
                    raise RuntimeError("Unknown reward model")

            # If the score is lower the better, reverse the scores
            if script_args.score_model_reverse:
                scores = [-score for score in scores]
            # Pack the scores into the result dictionary
            for idx, score in zip(indices, scores):
                if idx not in result:
                    result[idx] = []
                result[idx].append(score)
            if accelerator.is_local_main_process:
                # Simulate the actual update by times the number of processes
                pbar.update(accelerator.num_processes)
        # Transform to list of dicts, otherwise gather_object() will not collect correctly
        result = [result]

    gathered = gather_object(result)
    results = {k: v for d in gathered for k, v in d.items()}

    # Add or replace the 'reward_score' column in the dataset
    response_dataset = (
        response_dataset.remove_columns("reward_score")
        if "reward_score" in response_dataset.column_names
        else response_dataset
    )
    response_dataset = response_dataset.add_column(
        "reward_score",
        [
            np.mean(results[idx]) if idx in results else None
            for idx in range(len(response_dataset))
        ],
    )
    return response_dataset


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Find the reward model by dataset type
    for dataset_prefix in REWARD_CONFIGS.keys():
        if script_args.run_name.split("_")[2].startswith(dataset_prefix):
            script_args.score_model_id = REWARD_CONFIGS[dataset_prefix]["id"]
            script_args.score_model_reverse = REWARD_CONFIGS[dataset_prefix]["reverse"]
            break

    # Model & Tokenizer
    model, tokenizer = load_score_model(script_args)
    model.eval()

    # Dataset
    response_dataset = load_generated_dataset(script_args)

    # Evaluation
    response_dataset = evaluate_reward(model, tokenizer, response_dataset, script_args)

    # Push to Hub
    DatasetDict(
        {"default": response_dataset},
    ).push_to_hub(
        HUGGINGFACE_CONFIGS["prefix"]["evaluations"] + script_args.run_name,
        script_args.tag,
    )


if __name__ == "__main__":
    main()

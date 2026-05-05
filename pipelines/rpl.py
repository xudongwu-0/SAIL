"""Online Robust Listwise DPO (RPL) entry point.

Usage::

    accelerate launch --num_processes=6 pipelines/rpl.py \
        --model Q0.5B --dataset U10 --tag tag1 \
        --beta 0.1 --rho 0.05 --K 4 \
        --num_train_steps 50 \
        --per_device_prompt_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-6

Outputs an adapter checkpoint under ``cache/checkpoints/<run>_<host>/`` and,
optionally, pushes to the HF Hub. By default *no* Hub push is performed (set
``--push_to_hub`` to enable); WandB is disabled by default.

Important design notes (do not delete):

* The dataset only needs a ``prompt`` column. We re-use the SAIL-format DPO
  datasets (which carry prompt/chosen/rejected) and ignore the static labels.
* The reference policy is the same model with the LoRA adapter disabled.
  We start from the *base* model + a freshly initialised LoRA adapter, so at
  step 0 ``g_theta == 0`` (sanity check §6.5 of the implementation guide).
* The reward model produces preference labels on the fly. Default RM is the
  one paired with the dataset in ``configs/models/reward.yaml``.
"""

from __future__ import annotations

import os
import sys
import shutil
import hashlib
import socket
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safe_rlhf.models import AutoModelForScore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cdpo import OnlineRobustListwiseDPOTrainer
from utils import CONFIGS, format_run_name, wandb_init


HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
MODEL_CONFIGS = CONFIGS.models.names
LORA_MODULES = CONFIGS.models.lora_modules
REWARD_CONFIGS = CONFIGS.models.reward
CACHE_CONFIGS = CONFIGS.utils.cache
HASHCODE = hashlib.sha1(socket.gethostname().encode()).hexdigest()[:8]

accelerator = Accelerator()
transformers.logging.set_verbosity_error()

# `accelerator.local_process_index` triggers a lazy state init that explodes
# when the script is launched without `accelerate launch` (e.g. plain
# `python pipelines/rpl.py` for a single-GPU smoke run). Read the rank
# directly from the env vars set by the launcher; fall back to 0.
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))


@dataclass
class ScriptArguments:
    model: str = field(metadata={"help": "model name (key in models/names.yaml)"})
    dataset: str = field(metadata={"help": "dataset name (key in datasets/preprocess.yaml)"})
    tag: str = field(metadata={"help": "experiment tag"})

    # Online RPL hyperparameters
    beta: float = field(default=0.1, metadata={"help": "DPO temperature"})
    rho: float = field(default=0.05, metadata={"help": "robustness coefficient (0..1)"})
    noise_eta: float = field(
        default=0.0,
        metadata={
            "help": "label-noise rate eta in [0,1]: per example, with prob eta"
            " replace the RM-derived ranking with a uniform random permutation"
            " (stress-tests the robust term)."
        },
    )
    K: int = field(default=4, metadata={"help": "candidates generated per prompt"})

    # Online score-function correction (full online gradient, math doc §9).
    lambda_sf: float = field(
        default=0.0,
        metadata={
            "help": "weight on the REINFORCE-style score-function term;"
            " 0 = direct-loss surrogate (current default), 1 = full online"
            " gradient. With K=2, rho=0, lambda_sf=1 this matches the"
            " online pairwise DPO targeted by SAIL DPR."
        },
    )
    sf_baseline: str = field(
        default="mean",
        metadata={"help": "score-function baseline: 'mean' or 'none'."},
    )

    # Generation
    max_prompt_length: int = field(default=512)
    max_target_length: int = field(default=128)
    gen_temperature: float = field(default=0.9)
    gen_top_p: float = field(default=0.95)

    # Optim / sizes
    num_train_steps: int = field(default=50)
    per_device_prompt_batch_size: int = field(default=1)
    per_device_evalreward_batch_size: int = field(default=4)

    # Eval
    n_eval_prompts: int = field(default=64)
    do_final_eval: bool = field(default=True)

    # Model loading
    use_lora: bool = field(default=True)
    use_q_lora: bool = field(default=False)
    use_flash_attn: bool = field(default=True)
    sft_tag: Optional[str] = field(
        default=None,
        metadata={
            "help": "If set, load an existing SFT LoRA adapter from the Hub at"
            " this revision. Otherwise start from the base model with a fresh"
            " LoRA adapter (recommended for the first online RPL runs)."
        },
    )

    # Bookkeeping
    push_to_hub_after_train: bool = field(
        default=False,
        metadata={"help": "Push the final adapter to the Hub after training."},
    )

    model_cache_dir: str = field(default=CACHE_CONFIGS["model_cache_dir"])
    dataset_cache_dir: str = field(default=CACHE_CONFIGS["dataset_cache_dir"])


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="none")
    bf16: bool = field(default=True)
    learning_rate: float = field(default=5.0e-6)
    warmup_steps: int = field(default=5)
    gradient_accumulation_steps: int = field(default=4)
    gradient_checkpointing: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    lr_scheduler_type: str = field(default="cosine")
    logging_first_step: bool = field(default=True)
    logging_steps: int = field(default=1)
    save_strategy: str = field(default="no")
    evaluation_strategy: str = field(default="no")
    remove_unused_columns: bool = field(default=False)
    report_to: str = field(default="none")
    seed: int = field(default=42)
    dataloader_num_workers: int = field(default=0)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_policy(script_args: ScriptArguments) -> tuple:
    base_id = MODEL_CONFIGS[script_args.model]
    compute_dtype = torch.bfloat16

    config = AutoConfig.from_pretrained(
        base_id,
        cache_dir=script_args.model_cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    base_model = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=compute_dtype,
        device_map={"": LOCAL_RANK},
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if script_args.use_q_lora
            else None
        ),
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if script_args.use_flash_attn else None,
        use_cache=False,
        cache_dir=script_args.model_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_id,
        use_fast=True,
        cache_dir=script_args.model_cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if script_args.model.startswith("Q") and tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
    # Left-pad so generation works correctly with batched prompts.
    tokenizer.padding_side = "left"

    if script_args.sft_tag is not None:
        # Reuse an existing SFT adapter from the Hub.
        peft_model_id = HUGGINGFACE_CONFIGS["prefix"]["models"] + format_run_name(
            pipeline="SFT",
            model=script_args.model,
            dataset=script_args.dataset,
            extra_params={},
        )
        model = PeftModel.from_pretrained(
            base_model,
            peft_model_id,
            revision=script_args.sft_tag,
            is_trainable=True,
            adapter_name="default",
            cache_dir=script_args.model_cache_dir,
        )
        # Load the same SFT adapter again, frozen, as the *reference*
        # adapter so the trainer can compute g_θ relative to SFT (rather
        # than relative to the bare base model). The OnlineRobust trainer
        # picks this up via ``ref_adapter_name="reference"``.
        model.load_adapter(
            peft_model_id,
            revision=script_args.sft_tag,
            adapter_name="reference",
            cache_dir=script_args.model_cache_dir,
        )
    else:
        # Fresh LoRA adapter (g_theta == 0 at step 0).
        target_modules = LORA_MODULES[script_args.model[0]]
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(base_model, lora_config)

    model.print_trainable_parameters()
    return model, tokenizer


def load_reward_model(script_args: ScriptArguments) -> tuple:
    """Pick the RM bound to the dataset prefix in ``configs/models/reward.yaml``."""
    rm_cfg = None
    for prefix in REWARD_CONFIGS.keys():
        if script_args.dataset.startswith(prefix):
            rm_cfg = REWARD_CONFIGS[prefix]
            break
    if rm_cfg is None:
        raise ValueError(
            f"No reward model configured for dataset prefix of {script_args.dataset!r}"
        )
    rm_id = rm_cfg["id"]
    rm_reverse = rm_cfg["reverse"]
    dev = {"": LOCAL_RANK}

    if rm_id.startswith("PKU-Alignment"):
        rm = AutoModelForScore.from_pretrained(
            rm_id,
            torch_dtype=torch.bfloat16,
            device_map=dev,
            cache_dir=script_args.model_cache_dir,
        )
        rm_tok = AutoTokenizer.from_pretrained(
            rm_id,
            padding_side="left",
            use_fast=True,
            cache_dir=script_args.model_cache_dir,
        )
    elif rm_id.startswith("openbmb"):
        rm = AutoModel.from_pretrained(
            rm_id,
            torch_dtype=torch.bfloat16,
            device_map=dev,
            trust_remote_code=True,
            use_cache=True,
            cache_dir=script_args.model_cache_dir,
        )
        rm_tok = AutoTokenizer.from_pretrained(
            rm_id,
            padding_side="left",
            use_fast=True,
            cache_dir=script_args.model_cache_dir,
        )
    else:
        raise ValueError(f"Unknown reward model {rm_id}")
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    if rm_tok.pad_token is None:
        rm_tok.pad_token = rm_tok.eos_token
    return rm, rm_tok, rm_id, rm_reverse


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_prompt_dataset(script_args: ScriptArguments):
    ds = load_dataset(
        HUGGINGFACE_CONFIGS["prefix"]["datasets"] + script_args.dataset,
        cache_dir=script_args.dataset_cache_dir,
    )
    train = ds["train"].remove_columns(
        [c for c in ds["train"].column_names if c != "prompt"]
    )
    eval_ = ds["eval"].remove_columns(
        [c for c in ds["eval"].column_names if c != "prompt"]
    )
    if script_args.n_eval_prompts > 0:
        eval_ = eval_.select(range(min(script_args.n_eval_prompts, len(eval_))))
    return train, eval_


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    extra_params = {
        "beta": script_args.beta,
        "rho": script_args.rho,
        "K": script_args.K,
    }
    run = format_run_name(
        pipeline="RPL",
        model=script_args.model,
        dataset=script_args.dataset,
        extra_params=extra_params,
    )
    # Encode noise_eta as a suffix on the run name so that cache paths and
    # HF revision names disambiguate noise-injection runs without polluting
    # the canonical RPL run-name schema.
    if script_args.noise_eta > 0.0:
        run = f"{run}_eta{script_args.noise_eta:.2f}"
    if script_args.lambda_sf > 0.0:
        run = f"{run}_sf{script_args.lambda_sf:.2f}"
    training_args.output_dir = os.path.join(
        CACHE_CONFIGS["checkpoint_cache_dir"], f"{run}_{HASHCODE}"
    )
    training_args.run_name = f"{script_args.tag}-{run}"
    training_args.hub_model_id = (
        HUGGINGFACE_CONFIGS["prefix"]["models"] + run
        if script_args.push_to_hub_after_train
        else None
    )
    training_args.push_to_hub = script_args.push_to_hub_after_train
    # Wire batch size knobs.
    training_args.per_device_train_batch_size = script_args.per_device_prompt_batch_size
    training_args.max_steps = script_args.num_train_steps

    if accelerator.is_local_main_process:
        print(f"[rpl] run = {run}")
        print(f"[rpl] output_dir = {training_args.output_dir}")
        if training_args.report_to and training_args.report_to != "none":
            wandb_init(run, script_args, training_args)

    model, tokenizer = load_policy(script_args)
    rm, rm_tok, rm_id, rm_reverse = load_reward_model(script_args)
    train_ds, eval_ds = load_prompt_dataset(script_args)

    trainer = OnlineRobustListwiseDPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        beta=script_args.beta,
        rho=script_args.rho,
        K=script_args.K,
        noise_eta=script_args.noise_eta,
        ref_adapter_name=("reference" if script_args.sft_tag is not None else None),
        reward_model=rm,
        reward_tokenizer=rm_tok,
        reward_model_id=rm_id,
        reward_model_reverse=rm_reverse,
        per_device_evalreward_batch_size=script_args.per_device_evalreward_batch_size,
        max_prompt_length=script_args.max_prompt_length,
        max_target_length=script_args.max_target_length,
        gen_temperature=script_args.gen_temperature,
        gen_top_p=script_args.gen_top_p,
        lambda_sf=script_args.lambda_sf,
        sf_baseline=script_args.sf_baseline,
    )

    trainer.train()

    # Final pairwise win-rate vs the (frozen-adapter) reference.
    if script_args.do_final_eval and accelerator.is_local_main_process:
        eval_prompts = list(eval_ds["prompt"])
        metrics = trainer.evaluate_pairwise_win_rate(
            eval_prompts, per_device_batch_size=script_args.per_device_prompt_batch_size
        )
        print(f"[rpl] final pairwise eval: {metrics}")
        # Write a row summary that is easy to grep / aggregate.
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, "summary.txt"), "w") as f:
            f.write(f"run={run}\n")
            f.write(f"beta={script_args.beta} rho={script_args.rho} K={script_args.K}"
                    f" eta={script_args.noise_eta}\n")
            for k, v in metrics.items():
                f.write(f"{k}={v}\n")

    if accelerator.is_local_main_process:
        save_path = os.path.join(training_args.output_dir, "final_adapter")
        trainer.model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"[rpl] saved adapter to {save_path}")
        if script_args.push_to_hub_after_train:
            # Push the PEFT adapter directly (transformers 5.x's
            # ``trainer.push_to_hub(revision=...)`` no longer accepts ``revision``;
            # the underlying ``create_model_card`` rejects the kwarg).
            try:
                repo_id = training_args.hub_model_id
                tag = script_args.tag
                from huggingface_hub import HfApi
                hf_api = HfApi(token=HUGGINGFACE_CONFIGS["token"])
                hf_api.create_repo(repo_id, exist_ok=True, repo_type="model")
                try:
                    hf_api.create_branch(repo_id=repo_id, branch=tag, exist_ok=True)
                except Exception as e:
                    print(f"[rpl] branch create warn: {e}")
                trainer.model.push_to_hub(
                    repo_id, revision=tag,
                    token=HUGGINGFACE_CONFIGS["token"],
                    commit_message=f"rpl run {run}",
                )
                tokenizer.push_to_hub(
                    repo_id, revision=tag, token=HUGGINGFACE_CONFIGS["token"],
                )
                print(f"[rpl] pushed adapter to {repo_id}@{tag}")
            except Exception as e:
                print(f"[rpl] push_to_hub failed (adapter still saved locally): {e}")


if __name__ == "__main__":
    main()

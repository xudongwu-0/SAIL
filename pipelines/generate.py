import os
import sys
from dataclasses import dataclass, field
from tqdm.auto import tqdm
import torch
from huggingface_hub import HfApi, add_collection_item
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from peft import PeftModel
from accelerate import Accelerator
from accelerate.utils import gather_object

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CONFIGS

HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
HFAPI = HfApi(HUGGINGFACE_CONFIGS["token"])
MODEL_CONFIGS = CONFIGS.models.names
DATASET_CONFIGS = CONFIGS.datasets.preprocess
CACHE_CONFIGS = CONFIGS.utils.cache

accelerator = Accelerator()
tqdm.pandas()
transformers.logging.set_verbosity_error()


@dataclass
class ScriptArguments:
    run: str = field(
        metadata={"help": "run name to generate"},
    )
    tag: str = field(
        metadata={"help": "tag for the experiment"},
    )
    eval_limit: int = field(
        default=-1,
        metadata={"help": "limit the number of samples to evaluate"},
    )
    bf16: bool = field(default=True, metadata={"help": "use bfloat16"})
    fp16: bool = field(default=False, metadata={"help": "use float16"})
    model_max_length: int = field(
        default=2048,
        metadata={"help": "maximum sequence length for the model"},
    )
    per_device_generation_batch_size: int = field(
        default=4,
        metadata={"help": "batch size per device"},
    )
    # Use LoRA by default, full training not supported
    use_lora: bool = field(
        default=True, metadata={"help": "use LoRA by default, do not change"}
    )
    use_flash_attn: bool = field(default=True, metadata={"help": "use flash attention"})
    num_beams: int = field(
        default=1,
        metadata={"help": "number of beams for beam search"},
    )
    do_sample: bool = field(
        default=False,
        metadata={"help": "use sampling for generation"},
    )
    use_contrastive_search: bool = field(
        # Use contrastive search to improve the quality of small models
        default=True,
        metadata={"help": "use contrastive search for generation"},
    )
    penalty_alpha: float = field(
        default=0.6,
        metadata={"help": "repetition penalty"},
    )
    top_k: int = field(
        default=4,
        metadata={"help": "top k tokens to sample from"},
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
def load_and_format_dataset(script_args):
    dataset = load_dataset(
        HUGGINGFACE_CONFIGS["prefix"]["datasets"] + script_args.run.split("_")[2],
        cache_dir=script_args.dataset_cache_dir,
    )
    select_columns = ["prompt", "chosen", "rejected"]
    eval_dataset = dataset["eval"].map(
        lambda sample: {col: sample[col] for col in select_columns},
        remove_columns=[
            col for col in dataset["eval"].column_names if col not in select_columns
        ],
        num_proc=4,
    )
    return eval_dataset


# Load model and tokenizer for inference
def load_and_config_model(script_args):
    config = AutoConfig.from_pretrained(
        MODEL_CONFIGS[script_args.run.split("_")[1]],
        cache_dir=script_args.model_cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False
    compute_dtype = (
        torch.float16
        if script_args.fp16
        else (torch.bfloat16 if script_args.bf16 else torch.float32)
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIGS[script_args.run.split("_")[1]],
        torch_dtype=compute_dtype,
        device_map={"": accelerator.local_process_index},
        # Update transformers package to >=4.38.0 and no need to use trust_remote_code
        trust_remote_code=True,
        # Use flash attention if specified
        attn_implementation="flash_attention_2" if script_args.use_flash_attn else None,
        use_cache=True,
        cache_dir=script_args.model_cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CONFIGS[script_args.run.split("_")[1]],
        model_max_length=script_args.model_max_length,
        padding_side="left",
        use_fast=True,
        cache_dir=script_args.model_cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token

    def _parse_run(run: str):
        # e.g. "DPR_L8B_U10_beta0.10g0.30gamma0.30"
        parts = run.split("_")
        algo = parts[0]      # DPR / DPO / SFT ...
        model = parts[1]     # L8B / Q0.5B ...
        dataset = parts[2]   # U10 ...
        return algo, model, dataset

    if script_args.use_lora:
        algo, model_name, dataset_name = _parse_run(script_args.run)

        # 1) Always load the target adapter (DPR/DPO/SFT itself)
        target_peft_id = HUGGINGFACE_CONFIGS["prefix"]["models"] + script_args.run

        # 2) If it's DPR/DPO, preload SFT adapter first (same tag) to avoid "DPR-only" instability
        if algo in ["DPR", "DPO"]:
            sft_run = f"SFT_{model_name}_{dataset_name}"
            sft_peft_id = HUGGINGFACE_CONFIGS["prefix"]["models"] + sft_run

            # ---- BASE + SFT ----
            model = PeftModel.from_pretrained(
                base_model,
                sft_peft_id,
                revision=script_args.tag,
                is_trainable=False,
                cache_dir=script_args.model_cache_dir,
            )

            # ---- (BASE+SFT) + DPR/DPO ----
            model = PeftModel.from_pretrained(
                model,
                target_peft_id,
                revision=script_args.tag,
                is_trainable=False,
                cache_dir=script_args.model_cache_dir,
            )
        else:
            # e.g. SFT-only generation
            model = PeftModel.from_pretrained(
                base_model,
                target_peft_id,
                revision=script_args.tag,
                is_trainable=False,
                cache_dir=script_args.model_cache_dir,
            )


    return model, tokenizer


def generate_responses(model, tokenizer, eval_dataset, script_args):
    # Progress bar only on main process
    pbar = tqdm(total=len(eval_dataset), disable=not accelerator.is_local_main_process)
    # Split the list of prompts to each process, note it only works for list
    all_prompts = list(eval_dataset["prompt"])
    with accelerator.split_between_processes(all_prompts) as process_prompts:
        dataloader = DataLoader(
            process_prompts,
            batch_size=script_args.per_device_generation_batch_size,
            # This is required to maintain the order of the prompts
            shuffle=False,
        )
        result = {"response": []}
        with torch.no_grad():
            for prompts in dataloader:
                inputs = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="longest",
                    truncation=False,
                    pad_to_multiple_of=8,
                    add_special_tokens=False,
                )
                generate_kwargs = {
                    "penalty_alpha": script_args.penalty_alpha,
                    "top_k": script_args.top_k,
                }
                outputs = model.generate(
                    **inputs.to(model.device),
                    num_beams=script_args.num_beams,
                    do_sample=script_args.do_sample,
                    max_length=script_args.model_max_length,
                    max_new_tokens=None,
                    pad_token_id=tokenizer.eos_token_id,
                    **generate_kwargs if script_args.use_contrastive_search else {},
                )
                responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                responses = [
                    response[len(prompt) :]
                    for prompt, response in zip(prompts, responses)
                ]
                result["response"].extend(responses)
                if accelerator.is_local_main_process:
                    # Simulate the actual update by times the number of processes
                    pbar.update(len(prompts) * accelerator.num_processes)
        # Transform to list of dicts, otherwise gather_object() will not collect correctly
        result = [result]
    gathered = gather_object(result)
    return gathered


def main():
    parser = HfArgumentParser(ScriptArguments)
    (script_args,) = parser.parse_args_into_dataclasses()

    # Model & Tokenizer
    model, tokenizer = load_and_config_model(script_args)
    model.eval()

    # Dataset
    eval_dataset = load_and_format_dataset(script_args)
    if script_args.eval_limit > 0:
        eval_dataset = eval_dataset.select(range(script_args.eval_limit))

    # Generation
    results = generate_responses(model, tokenizer, eval_dataset, script_args)

    # Push to Hub
    if accelerator.is_local_main_process:
        response_dataset = eval_dataset.add_column(
            "response",
            [response for result in results for response in result["response"]],
        )
        DatasetDict(
            {"default": response_dataset},
        ).push_to_hub(
            HUGGINGFACE_CONFIGS["prefix"]["evaluations"] + script_args.run,
            script_args.tag,
        )
        add_collection_item(
            collection_slug=HUGGINGFACE_CONFIGS["collections"]["evaluations"],
            item_id=HUGGINGFACE_CONFIGS["prefix"]["evaluations"] + script_args.run,
            item_type="dataset",
            exists_ok=True,
        )


if __name__ == "__main__":
    main()

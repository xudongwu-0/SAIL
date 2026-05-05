"""
Push the local fresh-init RPL ρ-sweep checkpoints to the HuggingFace Hub so
that the original SAIL `cdpo gen / evalreward / evalgpt` pipelines can
target them.

Each adapter is pushed to::

    xudongwu/<run_name>             (revision = `freshinit`)

so that, e.g., ::

    cdpo gen -p RPL -m Q0.5B -d U10 -t freshinit \
        --beta 0.1 --rho 0.05 --K 4 -g rtxa6000:6

resolves to ``xudongwu/RPL_Q0.5B_U10_beta0.10rho0.05K4@freshinit``.

Run on CPU; this is purely IO.

Usage: python pipelines/push_freshinit_to_hub.py [--rho 0.05]
"""

import os
import sys
import glob
from dataclasses import dataclass, field
from typing import Optional

from huggingface_hub import HfApi, create_repo
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CONFIGS

HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
MODEL_CONFIGS = CONFIGS.models.names
CACHE_CONFIGS = CONFIGS.utils.cache
HFAPI = HfApi(token=HUGGINGFACE_CONFIGS["token"])


@dataclass
class Args:
    base_model: str = field(default="Q0.5B")
    rho: Optional[float] = field(
        default=None,
        metadata={"help": "If set, only push that single ρ; otherwise push all 5."},
    )
    revision: str = field(default="freshinit")
    checkpoint_glob: str = field(
        default="cache/checkpoints/RPL_Q0.5B_U10_beta0.10rho*K4_*",
    )


def push_one(adapter_dir: str, run_name: str, base_id: str, revision: str):
    repo_id = HUGGINGFACE_CONFIGS["prefix"]["models"] + run_name
    print(f"[push] {adapter_dir}  ->  {repo_id}@{revision}")

    create_repo(repo_id, token=HUGGINGFACE_CONFIGS["token"], exist_ok=True,
                repo_type="model")
    # Ensure the branch exists.
    try:
        HFAPI.create_branch(repo_id=repo_id, branch=revision, exist_ok=True)
    except Exception as e:
        print(f"  branch create warn: {e}")

    # Re-load on CPU to attach base for proper push.
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        cache_dir=CACHE_CONFIGS["model_cache_dir"], trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir, adapter_name="default")
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True,
                                         cache_dir=CACHE_CONFIGS["model_cache_dir"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model.push_to_hub(repo_id, revision=revision, token=HUGGINGFACE_CONFIGS["token"],
                       commit_message=f"upload fresh-init RPL adapter {run_name}")
    tok.push_to_hub(repo_id, revision=revision, token=HUGGINGFACE_CONFIGS["token"])
    print(f"  ✓ pushed {repo_id}@{revision}")
    del model, base
    import gc; gc.collect()


def main():
    args = HfArgumentParser(Args).parse_args_into_dataclasses()[0]
    base_id = MODEL_CONFIGS[args.base_model]
    dirs = sorted(glob.glob(args.checkpoint_glob))
    print(f"[push] found {len(dirs)} checkpoint(s)")

    for d in dirs:
        bn = os.path.basename(d)
        run_name = bn.rsplit("_", 1)[0]  # strip trailing _<HASHCODE>
        # rho parsing: name like RPL_Q0.5B_U10_beta0.10rho0.05K4 -> rho=0.05
        try:
            rho_str = run_name.split("rho")[1].split("K")[0]
            rho = float(rho_str)
        except Exception:
            print(f"  ! couldn't parse rho from {run_name}, skipping")
            continue
        if args.rho is not None and abs(rho - args.rho) > 1e-9:
            continue
        adapter_dir = os.path.join(d, "final_adapter")
        if not os.path.exists(adapter_dir):
            print(f"  ! no final_adapter in {d}, skipping")
            continue
        push_one(adapter_dir, run_name, base_id, args.revision)

    print("[push] done")


if __name__ == "__main__":
    main()

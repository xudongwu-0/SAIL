"""
Offline-style ranking accuracy on the U10 raw 4-way eval split.

For each held-out prompt, UltraFeedback provides four completions with an
``overall_score`` field.  We treat ``argsort(overall_score, descending)`` as
the ground-truth ranking, then ask: does the policy "know" this ordering?

Per-prompt computation
----------------------
    g_i  = beta * ( log p_policy(c_i | prompt) - log p_ref(c_i | prompt) )
                                                 i = 1..K (K = 4)
    sigma_pred  = argsort(g, descending)
    sigma_true  = argsort(overall_score, descending)

Reported metrics (mean over prompts):
    * top1_acc             -- fraction where sigma_pred[0] == sigma_true[0]
    * pairwise_acc         -- mean over K*(K-1)/2 pairs (i<j) of
                              1[sign(g_i - g_j) == sign(s_i - s_j)]  (ties skipped)
    * kendall_tau          -- scipy.stats.kendalltau averaged over prompts
    * ndcg                 -- standard NDCG@K with gain = max(score, 0)

Usage
-----
    # Reference policy = SFT adapter, evaluator policy = SFT adapter
    # (this gives g == 0 ⇒ random ranking baseline)
    python pipelines/eval_ranking.py \\
        --policy_repo xudongwu/SFT_Q0.5B_U10 --policy_revision Q0.5B \\
        --reference_repo xudongwu/SFT_Q0.5B_U10 --reference_revision Q0.5B \\
        --n_prompts 256 --tag sft_self

    # DPR baseline
    python pipelines/eval_ranking.py \\
        --policy_repo xudongwu/DPR_Q0.5B_U10_beta0.10g0.30gamma0.30 \\
        --policy_revision Q0.5B --tag dpr

    # RPL-from-SFT
    python pipelines/eval_ranking.py \\
        --policy_repo xudongwu/RPL_Q0.5B_U10_beta0.10rho0.05K4 \\
        --policy_revision Q0.5B --tag rpl_from_sft_rho0.05

    # A fresh-init RPL run (started from base, not SFT) -- override the reference
    python pipelines/eval_ranking.py \\
        --policy_path cache/checkpoints/RPL_Q0.5B_U10_beta0.10rho0.05K4_48625f66/final_adapter \\
        --reference_path none --tag rpl_freshinit_rho0.05

Pass ``--reference_path none`` (or ``--reference_repo none``) to use the
base-model log-probs as the reference (i.e. no adapter); this matches the
fresh-LoRA-init reference used inside the ρ-sweep training.

Single GPU, ~2-3 minutes for 256 prompts on Qwen1.5-0.5B.
"""

import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
)
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CONFIGS

HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
MODEL_CONFIGS = CONFIGS.models.names
DATASET_CONFIGS = CONFIGS.datasets.preprocess
CACHE_CONFIGS = CONFIGS.utils.cache

DATASET_SPLIT_SEED = 42  # mirror configs/datasets/preprocess seed


@dataclass
class ScriptArguments:
    model: str = field(default="Q0.5B")
    dataset: str = field(default="U10")
    beta: float = field(default=0.1)

    # Policy adapter (one of the two must be set)
    policy_repo: str = field(default="none")
    policy_revision: str = field(default="main")
    policy_path: str = field(default="none")

    # Reference adapter ("none" => use base model with no adapter)
    reference_repo: str = field(default="xudongwu/SFT_Q0.5B_U10")
    reference_revision: str = field(default="Q0.5B")
    reference_path: str = field(default="none")

    n_prompts: int = field(default=256)
    max_prompt_length: int = field(default=384)
    max_target_length: int = field(default=512)
    batch_size: int = field(default=4)
    tag: str = field(default="default")
    out_dir: str = field(default="results/ranking")

    model_cache_dir: str = field(default=CACHE_CONFIGS["model_cache_dir"])
    dataset_cache_dir: str = field(default=CACHE_CONFIGS["dataset_cache_dir"])


# ---------------------------------------------------------------------------
def load_eval_split(args: ScriptArguments):
    """Re-create the SAIL U10 eval split (seed=42, 0.2 ratio) with the raw
    4-completion structure preserved."""
    cfg = DATASET_CONFIGS[args.dataset]
    raw = load_dataset(
        cfg["id"], name=cfg["name"], split=cfg["split"][0],
        cache_dir=args.dataset_cache_dir,
    )
    if cfg["limit"]:
        raw = raw.select(range(cfg["limit"]))
    splitted = raw.train_test_split(test_size=cfg["ratio"], seed=DATASET_SPLIT_SEED)
    eval_ds = splitted["test"]
    # U10 filter: keep only samples with 4 completions (matches preprocess.py)
    eval_ds = eval_ds.filter(lambda s: len(s["completions"]) == 4, num_proc=4)
    return eval_ds


def load_models(args: ScriptArguments):
    base_id = MODEL_CONFIGS[args.model]
    config = AutoConfig.from_pretrained(base_id, trust_remote_code=True,
                                        cache_dir=args.model_cache_dir)
    config.use_cache = False

    def _make_base():
        return AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16,
            device_map={"": 0}, trust_remote_code=True, use_cache=False,
            cache_dir=args.model_cache_dir,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_id, use_fast=True,
                                              cache_dir=args.model_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.model.startswith("Q") and tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # right-pad for log-prob computation

    # ---- policy ----
    pol_base = _make_base()
    if args.policy_path != "none":
        policy = PeftModel.from_pretrained(pol_base, args.policy_path,
                                           adapter_name="default", is_trainable=False)
    elif args.policy_repo != "none":
        policy = PeftModel.from_pretrained(
            pol_base, args.policy_repo, revision=args.policy_revision,
            adapter_name="default", is_trainable=False,
            cache_dir=args.model_cache_dir,
        )
    else:
        raise ValueError("Must set policy_repo or policy_path")
    policy.eval()

    # ---- reference ----
    if args.reference_path != "none" and args.reference_path != "":
        ref_base = _make_base()
        reference = PeftModel.from_pretrained(ref_base, args.reference_path,
                                              adapter_name="default", is_trainable=False)
        reference.eval()
    elif args.reference_repo != "none" and args.reference_repo != "":
        ref_base = _make_base()
        reference = PeftModel.from_pretrained(
            ref_base, args.reference_repo, revision=args.reference_revision,
            adapter_name="default", is_trainable=False,
            cache_dir=args.model_cache_dir,
        )
        reference.eval()
    else:
        # Use base model with no adapter as reference.
        reference = _make_base()
        reference.eval()

    return policy, reference, tokenizer


# ---------------------------------------------------------------------------
@torch.no_grad()
def sequence_log_probs(model, tokenizer, prompts, responses,
                       max_prompt_length, max_target_length, device):
    """Sum of log p(response token | prompt + previous response tokens) per row.

    Both prompts and responses are length-B lists of strings."""
    B = len(prompts)
    prompt_ids = tokenizer(prompts, add_special_tokens=False, truncation=True,
                           max_length=max_prompt_length).input_ids
    resp_ids = tokenizer(responses, add_special_tokens=False, truncation=True,
                         max_length=max_target_length).input_ids
    full_ids = [p + r for p, r in zip(prompt_ids, resp_ids)]
    prompt_lens = [len(p) for p in prompt_ids]
    resp_lens = [len(r) for r in resp_ids]
    max_len = max(len(x) for x in full_ids)

    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((B, max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(full_ids):
        input_ids[i, :len(ids)] = torch.tensor(ids, device=device)
        attn[i, :len(ids)] = 1

    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits.float()  # (B, T, V)
    # Shift: predict token t from logits[t-1]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    tok_lp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # Mask: only response tokens contribute. Response token positions in input_ids
    # are [prompt_len .. prompt_len + resp_len - 1]; in shifted space (predict next)
    # the relevant log-probs live at indices [prompt_len-1 .. prompt_len + resp_len - 2].
    mask = torch.zeros_like(tok_lp, dtype=torch.bool)
    for i, (pl, rl) in enumerate(zip(prompt_lens, resp_lens)):
        if rl > 0 and pl >= 1:
            mask[i, pl - 1: pl - 1 + rl] = True
    sums = (tok_lp * mask).sum(dim=-1)
    return sums.cpu()


def kendall_tau(a, b):
    """Tau-b on small int ranks; returns NaN when degenerate."""
    from scipy.stats import kendalltau
    tau, _ = kendalltau(a, b)
    return tau if not np.isnan(tau) else 0.0


def ndcg_at_k(true_scores, pred_scores):
    """Standard NDCG with gain = max(score, 0). Order positions by pred."""
    order = np.argsort(-np.asarray(pred_scores))
    gains = np.maximum(np.asarray(true_scores)[order], 0.0)
    discounts = 1.0 / np.log2(np.arange(len(gains)) + 2)
    dcg = (gains * discounts).sum()
    ideal_gains = np.sort(np.maximum(true_scores, 0.0))[::-1]
    idcg = (ideal_gains * discounts).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
def main():
    args = HfArgumentParser(ScriptArguments).parse_args_into_dataclasses()[0]
    eval_ds = load_eval_split(args)
    n = min(args.n_prompts, len(eval_ds))
    eval_ds = eval_ds.select(range(n))
    print(f"[eval_ranking] evaluating on {n} prompts (K=4 each)")

    policy, reference, tokenizer = load_models(args)
    device = next(policy.parameters()).device

    top1_correct = 0
    pairwise_correct = 0
    pairwise_total = 0
    kendall_sum = 0.0
    ndcg_sum = 0.0
    g_means, g_stds = [], []

    for idx, sample in enumerate(eval_ds):
        prompt = f'[INST] {sample["instruction"]} [/INST] '
        comps = sample["completions"]
        if len(comps) != 4:
            continue
        responses = [c["response"] for c in comps]
        true_scores = np.array([float(c["overall_score"]) for c in comps])

        prompts_b = [prompt] * 4
        lp_pol = sequence_log_probs(
            policy, tokenizer, prompts_b, responses,
            args.max_prompt_length, args.max_target_length, device,
        )
        lp_ref = sequence_log_probs(
            reference, tokenizer, prompts_b, responses,
            args.max_prompt_length, args.max_target_length, device,
        )
        g = (args.beta * (lp_pol - lp_ref)).numpy()  # (4,)
        g_means.append(float(g.mean()))
        g_stds.append(float(g.std()))

        # Top-1 with tie-aware credit (random tie-break in expectation).
        true_top = int(np.argmax(true_scores))
        max_g = g.max()
        top_set = np.where(g >= max_g - 1e-12)[0]
        top1_correct += float(true_top in top_set) / len(top_set)

        # Pairwise (skip score ties on the *truth* side; give 0.5 on g ties).
        for i in range(4):
            for j in range(i + 1, 4):
                if true_scores[i] == true_scores[j]:
                    continue
                pairwise_total += 1
                diff = (g[i] - g[j]) * (true_scores[i] - true_scores[j])
                if abs(g[i] - g[j]) < 1e-12:
                    pairwise_correct += 0.5
                elif diff > 0:
                    pairwise_correct += 1

        kendall_sum += kendall_tau(true_scores, g)
        ndcg_sum += ndcg_at_k(true_scores, g)

        if (idx + 1) % 32 == 0:
            print(f"  [{idx+1}/{n}] running top1={top1_correct/(idx+1):.3f}"
                  f" pair={pairwise_correct/max(pairwise_total,1):.3f}"
                  f" tau={kendall_sum/(idx+1):.3f}"
                  f" ndcg={ndcg_sum/(idx+1):.3f}")

    metrics = {
        "n_prompts": n,
        "top1_acc": top1_correct / n,
        "pairwise_acc": pairwise_correct / max(pairwise_total, 1),
        "kendall_tau": kendall_sum / n,
        "ndcg": ndcg_sum / n,
        "g_mean_mean": float(np.mean(g_means)),
        "g_std_mean": float(np.mean(g_stds)),
    }
    print(f"[eval_ranking] {args.tag}: {json.dumps(metrics, indent=2)}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "metrics": metrics}, f, indent=2)
    print(f"[eval_ranking] wrote {out_path}")


if __name__ == "__main__":
    main()

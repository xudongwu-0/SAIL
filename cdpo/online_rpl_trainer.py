"""Online robust listwise DPO trainer.

Implements the **online** robust listwise DPO objective on top of HF's
``transformers.Trainer``. The math is the source of truth in
``vibecoding/ROBUST_LISTWISE_DPO_MATH.md`` and the implementation follows
``vibecoding/online_robust_listwise_dpo_implementation_guide.md``.

For each train step on a batch of prompts ``x_1..x_B``:

1. Sample ``K`` candidate responses per prompt with the *current* policy.
2. Score all ``B*K`` (prompt, response) pairs with a frozen reward model.
3. ``sigma_obs = argsort(rewards, descending)``.
4. ``g_theta = beta * (logp_policy - logp_ref)`` where the ref is the same
   model with the LoRA adapter disabled (guide §4.2).
5. Loss = ``(1 - rho) * PL(g, sigma_obs) + rho * PL(g, sigma_wc)``.

Online generation is the only stochastic source — preference labels come
*on the fly* from the reward model. Hence "online RLHF".
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models.utils import unwrap_model_for_generation

from .listwise_losses import (
    plackett_luce_loss,
    robust_pl_loss,
    worst_case_ranking,
)


if is_peft_available():
    from peft import PeftModel  # noqa: F401

if is_wandb_available():
    import wandb  # noqa: F401

if is_deepspeed_available():
    import deepspeed  # noqa: F401


def _per_sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_pad_token_id: int = -100,
) -> torch.Tensor:
    """``[N]`` sum of log-probs over response tokens (guide §4.3)."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    mask = (shift_labels != label_pad_token_id).to(log_probs.dtype)
    safe_labels = shift_labels.clamp(min=0).unsqueeze(-1)
    token_lp = log_probs.gather(-1, safe_labels).squeeze(-1) * mask
    return token_lp.sum(dim=1)


@contextmanager
def _disabled_adapter(model: nn.Module):
    """``disable_adapter()`` on the inner PEFT model (guide §4.2)."""
    base = model.module if hasattr(model, "module") else model
    with base.disable_adapter():
        yield


@contextmanager
def _switch_to_ref(model: nn.Module, ref_adapter_name: Optional[str]):
    """Make ``model`` produce log-probs / generations under the reference
    policy.  Two modes:

        * ``ref_adapter_name is None``  ⇒  disable all adapters (= base model).
        * ``ref_adapter_name`` set      ⇒  temporarily switch the active
          adapter to that name (e.g. a frozen "reference" SFT adapter loaded
          alongside the trainable "default").  Restores the previous active
          adapter on exit.
    """
    base = model.module if hasattr(model, "module") else model
    if ref_adapter_name is None:
        with base.disable_adapter():
            yield
        return
    # Save current active adapter(s) and switch.
    prev = getattr(base, "active_adapters", None)
    if callable(prev):
        prev = prev()
    elif prev is None:
        prev = getattr(base, "active_adapter", "default")
    try:
        base.set_adapter(ref_adapter_name)
        yield
    finally:
        # ``set_adapter`` accepts a single name or a list of names.
        try:
            if isinstance(prev, (list, tuple)):
                base.set_adapter(list(prev))
            else:
                base.set_adapter(prev or "default")
        except Exception:
            base.set_adapter("default")


class OnlineRobustListwiseDPOTrainer(Trainer):
    """Online robust listwise DPO trainer.

    Args (in addition to the standard ``Trainer`` args):
        ref_model: ignored. Reference is obtained either by disabling the
            adapter (default) or by switching to ``ref_adapter_name`` if set.
        ref_adapter_name: if not None, the reference policy is the PEFT
            adapter of that name (must already be loaded on the model, e.g.
            via ``model.load_adapter(..., adapter_name="reference")``).
            Use this to RPL-train **on top of an SFT adapter**, with SFT
            held frozen as the reference (so g_θ measures the *delta from
            SFT*, not the delta from the base model).
        beta: DPO temperature (defaults to 0.1).
        rho: robustness coefficient in [0, 1].
        K: number of candidates generated per prompt.
        reward_model / reward_tokenizer: frozen scorer (e.g. Eurus-RM-7b).
        reward_model_id: used to dispatch RM-specific output handling.
        reward_model_reverse: True if lower score = preferred.
        per_device_evalreward_batch_size: batch size for the RM forward.
        max_prompt_length / max_target_length: truncation lengths.
        gen_temperature / gen_top_p: sampling parameters.
        lambda_sf: weight on the score-function (REINFORCE-style) on-line
            gradient correction term. The full online robust gradient is
            ``∇L_rob + L_rob · Σ_i ∇log π_θ(y_i | x)`` (cf. math doc §9).
            Setting ``lambda_sf=0`` (default) gives the *direct* loss-only
            surrogate currently in use; setting it to ``1.0`` recovers the
            full online gradient. With ``K=2`` and ``rho=0`` this turns the
            objective into the online pairwise DPO that the SAIL DPR
            algorithm targets.
        sf_baseline: subtract this scalar from the per-example loss before
            multiplying by the log-prob (variance reduction). ``"mean"``
            (default) uses the batch mean as the baseline; ``"none"`` or
            ``0.0`` uses no baseline.
    """

    _tag_names = ["trl", "dpo", "online", "robust", "listwise"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        args: TrainingArguments,
        train_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        eval_dataset: Optional[Dataset] = None,
        callbacks=None,
        # ---- listwise / online specifics ----
        beta: float = 0.1,
        rho: float = 0.05,
        K: int = 4,
        noise_eta: float = 0.0,
        ref_adapter_name: Optional[str] = None,
        reward_model: Optional[nn.Module] = None,
        reward_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        reward_model_id: str = "",
        reward_model_reverse: bool = False,
        per_device_evalreward_batch_size: int = 4,
        max_prompt_length: int = 512,
        max_target_length: int = 256,
        gen_temperature: float = 0.9,
        gen_top_p: float = 0.95,
        label_pad_token_id: int = -100,
        lambda_sf: float = 0.0,
        sf_baseline: str = "mean",
    ) -> None:
        if reward_model is None or reward_tokenizer is None:
            raise ValueError(
                "OnlineRobustListwiseDPOTrainer requires a reward_model and"
                " reward_tokenizer for on-the-fly preference scoring."
            )
        if not (0.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        if not (0.0 <= noise_eta <= 1.0):
            raise ValueError(f"noise_eta must be in [0, 1], got {noise_eta}")
        if K < 2:
            raise ValueError(f"K must be >= 2, got {K}")
        if lambda_sf < 0.0:
            raise ValueError(f"lambda_sf must be >= 0, got {lambda_sf}")
        if sf_baseline not in ("mean", "none"):
            raise ValueError(
                f"sf_baseline must be 'mean' or 'none', got {sf_baseline!r}"
            )

        self.beta = beta
        self.rho = rho
        self.K = K
        self.noise_eta = noise_eta
        self.ref_adapter_name = ref_adapter_name
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.reward_model_id = reward_model_id
        self.reward_model_reverse = reward_model_reverse
        self.per_device_evalreward_batch_size = per_device_evalreward_batch_size
        self.max_prompt_length = max_prompt_length
        self.max_target_length = max_target_length
        self.gen_temperature = gen_temperature
        self.gen_top_p = gen_top_p
        self.label_pad_token_id = label_pad_token_id
        self.lambda_sf = lambda_sf
        self.sf_baseline = sf_baseline
        self._stored_metrics: Dict[str, Dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

        # The dataset only needs a "prompt" column. We pass through a no-op
        # collator that just stacks prompts into a list.
        if args.remove_unused_columns:
            args.remove_unused_columns = False

        super().__init__(
            model=model,
            args=args,
            data_collator=self._prompt_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )
        # PEFT integration check.
        self.is_peft_model = is_peft_available() and isinstance(
            self.accelerator.unwrap_model(self.model), PeftModel
        )
        if not self.is_peft_model:
            raise ValueError(
                "OnlineRobustListwiseDPOTrainer requires a PEFT model so we"
                " can derive pi_ref via disable_adapter(). Pass a PeftModel."
            )
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

    # ---------------------------------------------------------------------
    # Data path
    # ---------------------------------------------------------------------
    @staticmethod
    def _prompt_collator(features: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        return {"prompt": [f["prompt"] for f in features]}

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Override default _prepare_inputs to skip tensor-to-device on lists.
        return inputs

    # ---------------------------------------------------------------------
    # Generation
    # ---------------------------------------------------------------------
    def _generate_K_candidates(
        self, model: nn.Module, prompts: List[str]
    ) -> List[List[str]]:
        """Sample ``K`` responses per prompt; return list of K-lists of strings."""
        tok = self.tokenizer
        enc = tok(
            prompts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_prompt_length,
            add_special_tokens=False,
        ).to(self.accelerator.device)
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
            with torch.no_grad():
                out = unwrapped.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    do_sample=True,
                    temperature=self.gen_temperature,
                    top_p=self.gen_top_p,
                    max_new_tokens=self.max_target_length,
                    num_return_sequences=self.K,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
        # `out` shape: [B*K, L_total]. Need to strip the prompt portion.
        B = len(prompts)
        K = self.K
        responses_per_prompt: List[List[str]] = []
        for i in range(B):
            this_prompt_responses: List[str] = []
            for k in range(K):
                row = out[i * K + k]
                # Left-padded inputs: skip the padded prefix using attention.
                # Easier: re-decode after slicing past the original prompt
                # length. We use the *original tokenized* prompt length plus
                # left-pad offset to find the response start.
                # Since tokenizer is left-padded, the prompt occupies the
                # last `prompt_lens[i]` tokens of the input → response starts
                # at index `enc["input_ids"].shape[1]` of the output.
                resp_ids = row[enc["input_ids"].shape[1]:]
                text = tok.decode(resp_ids, skip_special_tokens=True)
                this_prompt_responses.append(text)
            responses_per_prompt.append(this_prompt_responses)
        return responses_per_prompt

    # ---------------------------------------------------------------------
    # Reward model scoring
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def _score_with_reward_model(
        self, prompts: List[str], responses_per_prompt: List[List[str]]
    ) -> torch.Tensor:
        """Return ``[B, K]`` reward tensor on CPU (float)."""
        flat_texts: List[str] = []
        for p, resps in zip(prompts, responses_per_prompt):
            for r in resps:
                flat_texts.append(p + r)
        scores: List[torch.Tensor] = []
        rm = self.reward_model
        rm_dev = next(rm.parameters()).device
        bs = self.per_device_evalreward_batch_size
        for i in range(0, len(flat_texts), bs):
            chunk = flat_texts[i : i + bs]
            enc = self.reward_tokenizer(
                chunk,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(rm_dev)
            out = rm(**enc)
            if self.reward_model_id.startswith("PKU-Alignment"):
                vals = out.end_scores
            elif self.reward_model_id.startswith("openbmb"):
                vals = out
            else:
                # Generic fallback: try .logits or use directly.
                vals = getattr(out, "logits", out)
            scores.append(vals.float().view(-1).cpu())
        flat_scores = torch.cat(scores, dim=0)
        if self.reward_model_reverse:
            flat_scores = -flat_scores
        return flat_scores.view(len(prompts), self.K)

    # ---------------------------------------------------------------------
    # g_theta
    # ---------------------------------------------------------------------
    def _build_concatenated_batch(
        self, prompts: List[str], responses_per_prompt: List[List[str]]
    ) -> Dict[str, torch.Tensor]:
        """Tokenize ``[B*K]`` (prompt + response) sequences with -100 mask on
        prompt and padding positions. Returns input_ids/attention_mask/labels.
        """
        tok = self.tokenizer
        flat_input_ids: List[List[int]] = []
        flat_labels: List[List[int]] = []
        for p, resps in zip(prompts, responses_per_prompt):
            p_ids = tok(p, add_special_tokens=False, truncation=True,
                        max_length=self.max_prompt_length)["input_ids"]
            for r in resps:
                r_ids = tok(r, add_special_tokens=False, truncation=True,
                            max_length=self.max_target_length)["input_ids"]
                # Append EOS so the model learns to terminate.
                if (
                    tok.eos_token_id is not None
                    and (len(r_ids) == 0 or r_ids[-1] != tok.eos_token_id)
                ):
                    r_ids = r_ids + [tok.eos_token_id]
                seq = p_ids + r_ids
                lab = [self.label_pad_token_id] * len(p_ids) + r_ids
                flat_input_ids.append(seq)
                flat_labels.append(lab)

        max_len = max(len(s) for s in flat_input_ids)
        pad_id = tok.pad_token_id
        N = len(flat_input_ids)
        input_ids = torch.full((N, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((N, max_len), dtype=torch.long)
        labels = torch.full((N, max_len), self.label_pad_token_id, dtype=torch.long)
        for i, (s, lab) in enumerate(zip(flat_input_ids, flat_labels)):
            L = len(s)
            input_ids[i, :L] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, :L] = 1
            labels[i, :L] = torch.tensor(lab, dtype=torch.long)
        device = self.accelerator.device
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        }

    def _compute_g(
        self, model: nn.Module, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(g, policy_lp, ref_lp)``; all ``[N]`` where ``N = B*K``."""
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        policy_lp = _per_sequence_log_probs(
            out.logits, batch["labels"], self.label_pad_token_id
        )
        with torch.no_grad(), _switch_to_ref(model, self.ref_adapter_name):
            ref_out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
        ref_lp = _per_sequence_log_probs(
            ref_out.logits.detach(), batch["labels"], self.label_pad_token_id
        )
        g = self.beta * (policy_lp - ref_lp)
        return g, policy_lp, ref_lp.detach()

    # ---------------------------------------------------------------------
    # Loss (HF Trainer hook)
    # ---------------------------------------------------------------------
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch=None,  # transformers >= 4.46 passes this; we ignore it
    ):
        prompts: List[str] = inputs["prompt"]
        B, K = len(prompts), self.K

        # 1. Generate K candidates per prompt with the current policy.
        responses = self._generate_K_candidates(model, prompts)

        # 2. Score with the frozen RM. Ranking 0 = best.
        rewards = self._score_with_reward_model(prompts, responses)  # [B, K] on CPU
        ranking_obs = torch.argsort(rewards, dim=1, descending=True).to(
            self.accelerator.device
        )

        # 2b. Optional label noise on σ_obs: with probability noise_eta per
        # example, replace the RM-derived ranking with a uniformly random
        # permutation. This stress-tests the robust term — the nominal PL
        # path now points at corrupted labels, while the worst-case path is
        # unaffected. With noise_eta=0 this is a no-op.
        n_corrupted = 0
        if self.noise_eta > 0.0:
            corrupt_mask = (
                torch.rand(B, device=ranking_obs.device) < self.noise_eta
            )
            if corrupt_mask.any():
                idx = corrupt_mask.nonzero(as_tuple=False).flatten().tolist()
                for i in idx:
                    perm = torch.randperm(K, device=ranking_obs.device)
                    ranking_obs[i] = perm
                n_corrupted = len(idx)

        # 3. g_theta on the [B*K] flat batch.
        flat_batch = self._build_concatenated_batch(prompts, responses)
        g_flat, policy_lp, ref_lp = self._compute_g(model, flat_batch)
        g = g_flat.view(B, K)

        # 4. Robust PL loss.
        loss_per_ex = robust_pl_loss(g, ranking_obs, self.rho, reduction="none")
        loss = loss_per_ex.mean()

        # 4b. Optional online score-function correction (math doc §9).
        # Full online gradient is
        #     ∇L_rob + L_rob · Σ_i ∇log π_θ(y_i | x).
        # We add a REINFORCE-style surrogate whose gradient is the second
        # term: ``stopgrad(L_rob_b - b(x)) · Σ_i log π_θ(y_{b,i} | x)``.
        sf_term_value = 0.0
        if self.lambda_sf > 0.0:
            # policy_lp is [B*K] for the generated responses; sum over K.
            sum_logp_per_prompt = policy_lp.view(B, K).sum(dim=1)  # [B]
            advantage = loss_per_ex.detach()
            if self.sf_baseline == "mean":
                # Use a cross-process baseline: per-device batch is usually
                # 1, so a local mean would zero out the advantage. Gathering
                # across DDP ranks gives a non-degenerate baseline.
                gathered = self.accelerator.gather(advantage)
                advantage = advantage - gathered.mean()
            sf_surrogate = (advantage * sum_logp_per_prompt).mean()
            loss = loss + self.lambda_sf * sf_surrogate
            sf_term_value = float(sf_surrogate.detach().item())

        # 5. Logging.
        with torch.no_grad():
            sigma_wc = worst_case_ranking(g)
            disagree = (sigma_wc != ranking_obs).any(dim=1).float().mean().item()
            metrics = {
                "g_mean": g.mean().item(),
                "g_std": g.std().item(),
                "rewards/mean": rewards.mean().item(),
                "rewards/best": rewards.max(dim=1).values.mean().item(),
                "rewards/worst": rewards.min(dim=1).values.mean().item(),
                "rewards/spread": (
                    rewards.max(dim=1).values - rewards.min(dim=1).values
                ).mean().item(),
                "ranking_disagreement_with_wc": disagree,
                "label_noise_frac": n_corrupted / max(B, 1),
                "policy_lp_mean": policy_lp.detach().mean().item(),
                "ref_lp_mean": ref_lp.mean().item(),
                "loss_rob_mean": float(loss_per_ex.detach().mean().item()),
                "sf_surrogate": sf_term_value,
            }
        self._store_metrics(metrics, "train")

        if return_outputs:
            return loss, {"g": g.detach(), "rewards": rewards}
        return loss

    # ---------------------------------------------------------------------
    # Eval (pairwise win rate vs reference)
    # ---------------------------------------------------------------------
    def evaluate_pairwise_win_rate(
        self,
        eval_prompts: List[str],
        per_device_batch_size: int = 2,
    ) -> Dict[str, float]:
        """Pairwise win rate of policy vs reference on a list of prompts.

        Both produce one greedy-ish sample per prompt; the RM picks the winner.
        """
        model = self.model
        tok = self.tokenizer

        def _generate(unwrapped: nn.Module, prompts: List[str]) -> List[str]:
            enc = tok(
                prompts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_prompt_length,
                add_special_tokens=False,
            ).to(self.accelerator.device)
            with torch.no_grad():
                out = unwrapped.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    do_sample=True,
                    temperature=self.gen_temperature,
                    top_p=self.gen_top_p,
                    max_new_tokens=self.max_target_length,
                    num_return_sequences=1,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            inp_len = enc["input_ids"].shape[1]
            return [tok.decode(o[inp_len:], skip_special_tokens=True) for o in out]

        wins, ties, losses = 0, 0, 0
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped:
            for i in range(0, len(eval_prompts), per_device_batch_size):
                chunk = eval_prompts[i : i + per_device_batch_size]
                pol_resp = _generate(unwrapped, chunk)
                with _switch_to_ref(unwrapped, self.ref_adapter_name):
                    ref_resp = _generate(unwrapped, chunk)
                rewards = self._score_with_reward_model_flat(
                    [p + r for p, r in zip(chunk, pol_resp)]
                    + [p + r for p, r in zip(chunk, ref_resp)]
                )
                n = len(chunk)
                pol_r = rewards[:n]
                ref_r = rewards[n:]
                wins += int((pol_r > ref_r).sum().item())
                ties += int((pol_r == ref_r).sum().item())
                losses += int((pol_r < ref_r).sum().item())
        total = max(wins + ties + losses, 1)
        return {
            "eval/pairwise_wins": wins,
            "eval/pairwise_ties": ties,
            "eval/pairwise_losses": losses,
            "eval/pairwise_win_rate": wins / total,
            "eval/pairwise_win_rate_with_ties": (wins + 0.5 * ties) / total,
        }

    @torch.no_grad()
    def _score_with_reward_model_flat(self, texts: List[str]) -> torch.Tensor:
        scores: List[torch.Tensor] = []
        rm = self.reward_model
        rm_dev = next(rm.parameters()).device
        bs = self.per_device_evalreward_batch_size
        for i in range(0, len(texts), bs):
            chunk = texts[i : i + bs]
            enc = self.reward_tokenizer(
                chunk,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(rm_dev)
            out = rm(**enc)
            if self.reward_model_id.startswith("PKU-Alignment"):
                vals = out.end_scores
            elif self.reward_model_id.startswith("openbmb"):
                vals = out
            else:
                vals = getattr(out, "logits", out)
            scores.append(vals.float().view(-1).cpu())
        flat = torch.cat(scores, dim=0)
        if self.reward_model_reverse:
            flat = -flat
        return flat

    # ---------------------------------------------------------------------
    # Logging plumbing
    # ---------------------------------------------------------------------
    def _store_metrics(
        self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        for k, v in metrics.items():
            self._stored_metrics[train_eval][k].append(float(v))

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        for k, vs in self._stored_metrics[train_eval].items():
            if vs:
                logs[k] = sum(vs) / len(vs)
        self._stored_metrics[train_eval].clear()
        if start_time is None:
            super().log(logs)
        else:
            super().log(logs, start_time)

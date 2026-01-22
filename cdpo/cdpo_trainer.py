# SAIL/cdpo/cdpo_trainer.py
import inspect
import random
import warnings
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from huggingface_hub import upload_folder, create_branch
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import (
    PreTrainedModelWrapper,
    create_reference_model,
)
from trl.models.utils import unwrap_model_for_generation
from accelerate import infer_auto_device_map, dispatch_model, cpu_offload
from accelerate.utils import gather_object, broadcast_object_list
from safe_rlhf.models import AutoModelForScore


from .trainer_utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)


if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


class GeneralizedDPOTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss. If no
            reference model is provided, the trainer will create a reference model with the same architecture as the model to be optimized.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy. For the IPO loss, beta is the regularization parameter denoted by tau in the paper.
        label_smoothing (`float`, defaults to 0):
            The robust DPO label smoothing parameter from the [cDPO](https://ericmitchell.ai/cdpo.pdf) report that should be between 0 and 0.5.
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model` and `ref_model`.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        compute_metrics (`Callable[[EvalPrediction], Dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        precompute_ref_log_probs (`bool`, defaults to `False`):
            Flag to precompute reference model log probabilities for training and evaluation datasets. This is useful if you want to train
            without the reference model and reduce the total GPU memory needed.
        dataset_num_proc (`Optional[int]`, *optional*):
            The number of workers to use to tokenize the data. Defaults to None.
        model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        ref_model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the ref model from a string
        model_adapter_name (`str`, defaults to `None`):
            Name of the train target PEFT adapter, when using LoRA with multiple adapters.
        ref_adapter_name (`str`, defaults to `None`):
            Name of the reference PEFT adapter, when using LoRA with multiple adapters.
        reference_free (`bool`):
            If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.
        force_use_ref_model (`bool`, defaults to `False`):
            In case one passes a PEFT model for the active model and you want to use a different model for the ref_model, set this flag to `True`.
    """

    _tag_names = ["trl", "dpo"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[Any] = None,
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        force_use_ref_model: bool = False,
        ##############################
        # New parameters
        loss_type: Literal[
            "sigmoid",
            "label_smoothing",
            "hinge",
            "ipo",
            "kto_pair",
            "generalized_sigmoid",
        ] = "generalized_sigmoid",
        generation_reuse_multiplier: Optional[int] = None,
        generation_num_batches: Optional[int] = None,
        per_device_generation_batch_size: Optional[int] = None,
        generation_temperature: Optional[float] = None,
        reward_model_id: Optional[str] = None,
        reward_model: Optional[AutoModelForScore] = None,
        reward_tokenizer: Optional[AutoTokenizer] = None,
        reward_model_reverse: Optional[bool] = None,
        per_device_evalreward_batch_size: Optional[int] = None,
        r: float = 0.0,
        rho: float = 0.0,
        p: float = 0.0,
        pi: float = 0.0,
        g: float = 0.0,
        gamma: float = 0.0,
        revkl: bool = False,
        revkl_coef: float = 0.0,
        revkl_on: str = "both",
        ##############################
    ):
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError(
                "You passed model_kwargs to the DPOTrainer. But your model is already instantiated."
            )

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the DPOTrainer. But your ref_model is already instantiated."
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(
                ref_model, **ref_model_init_kwargs
            )

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if ref_model is not None and not force_use_ref_model:
                raise ValueError(
                    "You passed both a ref_model and a peft_config. For training PEFT adapters with DPO there is no need to pass a reference"
                    " model. Please pass `ref_model=None` in case you want to train PEFT adapters, or pass a ref_model with `force_use_ref_model=True` in DPOTrainer's init."
                    " if you want to use a different ref_model."
                )

            if getattr(model, "is_loaded_in_8bit", False) or getattr(
                model, "is_loaded_in_4bit", False
            ):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {
                    "use_gradient_checkpointing": args.gradient_checkpointing
                }

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = (
                        args.gradient_checkpointing_kwargs
                    )

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(
                        make_inputs_require_grad
                    )

            # get peft model with the given config
            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                # If args.bf16 we need to explicitly call `generate` with torch amp autocast context manager
                self._peft_has_been_casted_to_bf16 = True

        # For models that use gradient_checkpoiting, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError(
                "When no model is provided, you need to pass the parameter is_encoder_decoder."
            )
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.model_adapter_name = model_adapter_name
        self.ref_adapter_name = ref_adapter_name
        self.reference_free = reference_free

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model or precompute_ref_log_probs:
            # The `model` with adapters turned off will be used as the reference model
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize a DPO dataset.")
        if max_length is None:
            warnings.warn(
                "`max_length` is not set in the DPOTrainer's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        if max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the DPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128

        if max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the DPOTrainer's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = (
            padding_value if padding_value is not None else tokenizer.pad_token_id
        )
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        self.beta = beta
        self.label_smoothing = label_smoothing

        ##############################
        # New parameters
        self.loss_type = loss_type
        self.generation_reuse_multiplier = generation_reuse_multiplier
        self.generation_num_batches = generation_num_batches
        self.per_device_generation_batch_size = per_device_generation_batch_size
        self.generation_temperature = generation_temperature
        self.reward_model_id = reward_model_id
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.reward_model_reverse = reward_model_reverse
        self.per_device_evalreward_batch_size = per_device_evalreward_batch_size
        self.r = r
        self.rho = rho
        self.p = p
        self.pi = pi
        self.g = g
        self.gamma = gamma
        # ===== RevKL (independent) =====
        self.revkl = revkl
        self.revkl_coef = revkl_coef
        self.revkl_on = revkl_on
        # ##############################
        self._dpp_generation_inputs = []
        self._dpp_generation_outputs = []
        self._dpr_generation_inputs = []
        self._dpr_generation_outputs = []
        self._ddp_sampling_mask = None
        self._dpp_sampling_mask = None
        self._dpr_sampling_mask = None
        ##############################

        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        self.dataset_num_proc = dataset_num_proc

        # Compute that only on the main process for faster data processing.
        # see: https://github.com/huggingface/trl/pull/1255
        with PartialState().local_main_process_first():
            # tokenize the dataset
            train_dataset = train_dataset.map(
                self.tokenize_row, num_proc=self.dataset_num_proc
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    self.tokenize_row, num_proc=self.dataset_num_proc
                )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if (
                self.accelerator.state.deepspeed_plugin.zero_stage == 3
                and self.precompute_ref_log_probs
            ):
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs):
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if (
                    hidden_size is not None
                    and config_kwargs["zero_optimization"]["stage"] == 3
                ):
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size
                            * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10
                            * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9
                            * hidden_size
                            * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(
                DataLoader(self.train_dataset, **dataloader_params)
            )

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(
                iterable=data_loader, desc="Train dataset reference log probs"
            ):
                reference_chosen_logp, reference_rejected_logp = (
                    self.compute_reference_log_probs(padded_batch)
                )
                reference_chosen_logp, reference_rejected_logp = (
                    self.accelerator.gather_for_metrics(
                        (reference_chosen_logp, reference_rejected_logp)
                    )
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = (
                torch.cat(reference_chosen_logps).float().numpy()
            )
            all_reference_rejected_logps = (
                torch.cat(reference_rejected_logps).float().numpy()
            )

            self.train_dataset = self.train_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(
                DataLoader(eval_dataset, **dataloader_params)
            )

            reference_chosen_logps = []
            reference_rejected_logps = []
            for padded_batch in tqdm(
                iterable=data_loader, desc="Eval dataset reference log probs"
            ):
                reference_chosen_logp, reference_rejected_logp = (
                    self.compute_reference_log_probs(padded_batch)
                )
                reference_chosen_logp, reference_rejected_logp = (
                    self.accelerator.gather_for_metrics(
                        (reference_chosen_logp, reference_rejected_logp)
                    )
                )
                reference_chosen_logps.append(reference_chosen_logp.cpu())
                reference_rejected_logps.append(reference_rejected_logp.cpu())

            all_reference_chosen_logps = (
                torch.cat(reference_chosen_logps).float().numpy()
            )
            all_reference_rejected_logps = (
                torch.cat(reference_rejected_logps).float().numpy()
            )

            eval_dataset = eval_dataset.add_column(
                name="reference_chosen_logps", column=all_reference_chosen_logps
            )
            eval_dataset = eval_dataset.add_column(
                name="reference_rejected_logps", column=all_reference_rejected_logps
            )

            # Save calculated reference_chosen_logps and reference_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][
            len(prompt_input_ids) :
        ]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError(
                "Prompt input ids and answer input ids should have the same length."
            )

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if (
            prompt_input_ids
            != full_tokenized["input_ids"][:response_token_ids_start_idx]
        ):
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][
            :response_token_ids_start_idx
        ]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError(
                "Prompt input ids and attention mask should have the same length."
            )

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][
            response_token_ids_start_idx:
        ]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(
        self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None
    ) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            # Last prompt token might get merged by tokenizer and
            # it should not be included for generation if that happens
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(
                chosen_prompt_len_input_ids, rejected_prompt_len_input_ids
            )

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Make sure prompts only have one different token at most an
            # and length only differs by 1 at most
            num_diff_tokens = sum(
                [
                    a != b
                    for a, b in zip(
                        chosen_tokens["prompt_input_ids"],
                        rejected_tokens["prompt_input_ids"],
                    )
                ]
            )
            num_diff_len = abs(
                chosen_prompt_len_input_ids - rejected_prompt_len_input_ids
            )
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )

            # add BOS token to head of prompt
            prompt_tokens["prompt_input_ids"] = [
                self.tokenizer.bos_token_id
            ] + prompt_tokens["prompt_input_ids"]
            chosen_tokens["prompt_input_ids"] = [
                self.tokenizer.bos_token_id
            ] + chosen_tokens["prompt_input_ids"]
            rejected_tokens["prompt_input_ids"] = [
                self.tokenizer.bos_token_id
            ] + rejected_tokens["prompt_input_ids"]

            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens[
                "prompt_attention_mask"
            ]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens[
                "prompt_attention_mask"
            ]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens[
                "prompt_attention_mask"
            ]

            # add EOS token to end of answer
            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(
                len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"])
            )

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if (
                    len(answer_tokens["prompt_input_ids"]) + longer_response_length
                    > self.max_length
                ):
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][
                                : self.max_prompt_length
                            ]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][
                                -self.max_prompt_length :
                            ]
                    else:
                        raise ValueError(
                            f"Unknown truncation mode: {self.truncation_mode}"
                        )

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if (
                    len(answer_tokens["prompt_input_ids"]) + longer_response_length
                    > self.max_length
                ):
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][
                            : self.max_length - self.max_prompt_length
                        ]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k]
                for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k]
                for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][
                : len(chosen_tokens["prompt_input_ids"])
            ] = [self.label_pad_token_id] * len(chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][
                :
            ]
            rejected_sequence_tokens["labels"][
                : len(rejected_tokens["prompt_input_ids"])
            ] = [self.label_pad_token_id] * len(rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen,
                truncation=True,
                max_length=self.max_target_length,
                add_special_tokens=True,
            )
            rejected_tokens = self.tokenizer(
                rejected,
                truncation=True,
                max_length=self.max_target_length,
                add_special_tokens=True,
            )
            prompt_tokens = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_prompt_length,
                add_special_tokens=True,
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(
                model, "prepare_decoder_input_ids_from_labels"
            ):
                batch["rejected_decoder_input_ids"] = (
                    model.prepare_decoder_input_ids_from_labels(
                        labels=torch.tensor(batch["rejected_labels"])
                    )
                )
                batch["chosen_decoder_input_ids"] = (
                    model.prepare_decoder_input_ids_from_labels(
                        labels=torch.tensor(batch["chosen_labels"])
                    )
                )

        return batch

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling null reference model (that is, peft adapter manipulation)."""
        with (
            self.accelerator.unwrap_model(self.model).disable_adapter()
            if self.is_peft_model and not self.ref_adapter_name
            else nullcontext()
        ):
            if self.ref_adapter_name:
                self.model.set_adapter(self.ref_adapter_name)
            yield
            if self.ref_adapter_name:
                self.model.set_adapter(self.model_adapter_name or "default")

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = (
            torch.cuda.amp.autocast
            if self._peft_has_been_casted_to_bf16
            else nullcontext
        )

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
            is_encoder_decoder: Whether the model is an encoder-decoder model.
            label_pad_token_id: The label pad token id.
            padding_value: The padding value to use for the concatenated inputs_ids.
            device: The device for the concatenated inputs.

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(
                batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1]
            )
        else:
            max_length = max(
                batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1]
            )

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(
                    batch[k], max_length, pad_value=pad_value
                )
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = (
                batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            )
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        if self.reference_free:
            ref_logratios = torch.tensor(
                [0], dtype=pi_logratios.dtype, device=pi_logratios.device
            )
        else:
            ref_logratios = reference_chosen_logps - reference_rejected_logps

        pi_logratios = pi_logratios.to(self.accelerator.device)
        ref_logratios = ref_logratios.to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        ##############################
        # DDP
        if train_eval == "train":
            # Probability of swithching the chosen and rejected responses
            # Which are independent Bernoulli random variables with probability 1 - \sigmoid(\beta * logits)
            policy_preference_switching_mask = (
                torch.bernoulli(1 - F.sigmoid(self.beta * logits))
                .bool()
                .to(logits.device)
            )
            # If both mixing and switching Bernoulli variables of a sample are 1, then the chosen and rejected responses are switched
            logits = (
                1 - 2 * self._ddp_sampling_mask * policy_preference_switching_mask
            ) * logits
        ##############################

        # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
        # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == "label_smoothing":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            # eqn (7) of the HALOs paper
            chosen_KL = (
                (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            )
            rejected_KL = (
                (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)
            )
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        ##############################
        # DDP & DPP
        elif self.loss_type == "generalized_sigmoid":
            # For the extra gradient term as (\nabla_\theta\logsigmoid(\beta * logits)) * \logsigmoid(\beta * logits), we do not need to modify the gradients
            # since the intergrated loss is just 1/2 * \logsigmoid(\beta * logits)^2
            losses = -F.logsigmoid(self.beta * logits)
            if train_eval == "train":
                losses -= (
                    0.5
                    * self.rho
                    * (F.logsigmoid(self.beta * logits) * self._ddp_sampling_mask) ** 2
                )
                losses -= (
                    0.5
                    * self.pi
                    * (F.logsigmoid(self.beta * logits) * self._dpp_sampling_mask) ** 2
                )
        ##############################
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
            )

        ##############################
        # DPP & DPR
        detached_loss = F.logsigmoid(self.beta * logits).detach()
        detached_chosen_logps = policy_chosen_logps.detach()
        detached_rejected_logps = policy_rejected_logps.detach()

        def chosen_logps_grad_hook(grad):
            return (
                grad
                - (
                    self.pi
                    * detached_loss
                    / detached_chosen_logps
                    * self._dpp_sampling_mask
                )
                - (
                    self.gamma
                    * detached_loss
                    / detached_chosen_logps
                    * self._dpr_sampling_mask
                )
            )

        def rejected_logps_grad_hook(grad):
            return (
                grad
                - (
                    self.pi
                    * detached_loss
                    / detached_rejected_logps
                    * self._dpp_sampling_mask
                )
                - (
                    self.gamma
                    * detached_loss
                    / detached_rejected_logps
                    * self._dpr_sampling_mask
                )
            )

        if train_eval == "train" and policy_chosen_logps.requires_grad:
            policy_chosen_logps.register_hook(chosen_logps_grad_hook)
        if train_eval == "train" and policy_rejected_logps.requires_grad:
            policy_rejected_logps.register_hook(rejected_logps_grad_hook)
        ##############################

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device)
                - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )

        return losses, chosen_rewards, rejected_rewards

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.
            is_encoder_decoder: Whether the model is an encoder-decoder model.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError(
                "Logits (batch and sequence length dim) and labels must have the same shape."
            )

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(
            logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    def _token_kl_ref_pi(self, ref_logits: torch.Tensor, pi_logits: torch.Tensor, labels: torch.Tensor):
        """
        Token-level proxy for reverse-KL: E_t[ log p_ref(y_t) - log p_pi(y_t) ]
        ref_logits, pi_logits: [B, T, V]
        labels: [B, T] with -100 for ignored tokens
        Returns: scalar tensor
        """
        # ---- 0) align lengths ----
        T = labels.size(1)
        ref_logits = ref_logits[:, :T, :]
        pi_logits  = pi_logits[:, :T, :]

        # ---- 1) build mask ----
        mask = (labels != -100)  # [B, T]
        safe_labels = labels.clone()
        safe_labels[~mask] = 0   # avoid gather out-of-range

        # ---- 2) logp(label) = logit(label) - logsumexp(logits) ----
        ref_tok_logit = torch.gather(ref_logits, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)  # [B,T]
        pi_tok_logit  = torch.gather(pi_logits,  dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1) # [B,T]

        ref_lse = torch.logsumexp(ref_logits, dim=-1)  # [B,T]
        pi_lse  = torch.logsumexp(pi_logits,  dim=-1)  # [B,T]

        logp_ref_tok = ref_tok_logit - ref_lse  # [B,T]
        logp_pi_tok  = pi_tok_logit  - pi_lse   # [B,T]

        # ---- 3) token avg over valid positions ----
        diff = (logp_ref_tok - logp_pi_tok) * mask  # [B,T]
        denom = mask.sum().clamp(min=1)
        return diff.sum() / denom



    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop(
                    "concatenated_decoder_input_ids", None
                ),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        ##############################
        # DDP & DPP & DPR
        if train_eval == "train":

            # Random state alreay synctronized and mask will be the same across all GPUs
            # Sampling masks for DDP & DPP & DPR
            probs = torch.tensor([self.r, self.p, self.g, 1 - self.r - self.p - self.g])
            sampling_routes = torch.multinomial(
                probs, len(batch["prompt"]), replacement=True
            )
            self._ddp_sampling_mask = (sampling_routes == 0).to(self.accelerator.device)
            self._dpp_sampling_mask = (sampling_routes == 1).to(self.accelerator.device)
            self._dpr_sampling_mask = (sampling_routes == 2).to(self.accelerator.device)

            # DPP & DPR
            if (self._dpp_sampling_mask | self._dpr_sampling_mask).sum() > 0:
                batch = self.generate_samples(model, batch)

        ##############################

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            train_eval,
        )

        # ===== RevKL regularizer (independent of DPR) =====
        if self.revkl and (self.revkl_coef > 0):
            # Compute reference logits (no grad). We already have policy logits from concatenated_forward.
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            ref_chosen_logits,
                            ref_rejected_logits,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        ref_chosen_logits,
                        ref_rejected_logits,
                    ) = self.concatenated_forward(self.ref_model, batch)
            parts = []
            if self.revkl_on in ("chosen", "both"):
                parts.append(self._token_kl_ref_pi(ref_chosen_logits, policy_chosen_logits, batch["chosen_labels"]))
            if self.revkl_on in ("rejected", "both"):
                parts.append(self._token_kl_ref_pi(ref_rejected_logits, policy_rejected_logits, batch["rejected_labels"]))

            revkl_loss = sum(parts) / len(parts)

            # losses is shape [B], revkl_loss is scalar -> broadcast add
            losses = losses + self.revkl_coef * revkl_loss
        # ===== End RevKL =====


    
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        if self.revkl and (self.revkl_coef > 0):
            metrics[f"{prefix}loss/revkl"] = revkl_loss.detach().cpu()
            metrics[f"{prefix}loss/revkl_coef"] = torch.tensor(self.revkl_coef).cpu()

        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (
            (chosen_rewards - rejected_rewards).mean().cpu()
        )
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = (
            policy_rejected_logits.detach().mean().cpu()
        )
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        compute_loss_context_manager = (
            torch.cuda.amp.autocast
            if self._peft_has_been_casted_to_bf16
            else nullcontext
        )

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(
                model, inputs, train_eval="train"
            )

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(
        self, model, batch: Dict[str, torch.LongTensor]
    ) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        # If one uses `generate_during_eval` with peft + bf16, we need to explictly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = (
            nullcontext
            if not self._peft_has_been_casted_to_bf16
            else torch.cuda.amp.autocast
        )

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                max_new_tokens=None,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            max_new_tokens=None,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        max_new_tokens=None,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

        policy_output = pad_to_length(
            policy_output, self.max_length, self.tokenizer.pad_token_id
        )
        policy_output_decoded = self.tokenizer.batch_decode(
            policy_output, skip_special_tokens=True
        )

        reference_output = pad_to_length(
            reference_output, self.max_length, self.tokenizer.pad_token_id
        )
        reference_output_decoded = self.tokenizer.batch_decode(
            reference_output, skip_special_tokens=True
        )

        return policy_output_decoded, reference_output_decoded

    ##############################
    # DPP & DPR
    def generate_samples(
        self, model, batch: Dict[str, torch.LongTensor]
    ) -> Dict[str, torch.LongTensor]:
        """Generate samples from the model for the given batch of prompts."""

        # Update the batch with the generated outputs
        if (len(self._dpp_generation_outputs) + len(self._dpr_generation_outputs)) > 0:
            updated_features = []
            for i, (prompt, chosen, rejected, dpp_m, dpr_m) in enumerate(
                zip(
                    batch["prompt"],
                    batch["chosen"],
                    batch["rejected"],
                    self._dpp_sampling_mask,
                    self._dpr_sampling_mask,
                )
            ):
                if dpp_m:
                    if len(self._dpp_generation_outputs) == 0:
                        warnings.warn("Insufficient generated responses for DPP")
                        self._dpp_sampling_mask[i].fill_(False)
                    else:
                        updated_features.append(self._dpp_generation_outputs.pop())
                        continue
                elif dpr_m:
                    if len(self._dpr_generation_outputs) == 0:
                        warnings.warn("Insufficient generated responses for DPR")
                        self._dpr_sampling_mask[i].fill_(False)
                    else:
                        updated_features.append(self._dpr_generation_outputs.pop())
                        continue
                updated_features.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                )
            # Replace the batch with the updated features
            batch = self.data_collator(
                [self.tokenize_row(f, model) for f in updated_features]
            )
            batch = {
                "prompt": [f["prompt"] for f in updated_features],
                "chosen": [f["chosen"] for f in updated_features],
                "rejected": [f["rejected"] for f in updated_features],
            } | batch
            batch = self._prepare_inputs(batch)

        # Collect the inputs for generation
        self._dpp_generation_inputs.extend(
            [p for p, dpp_m in zip(batch["prompt"], self._dpp_sampling_mask) if dpp_m]
        )
        self._dpr_generation_inputs.extend(
            [p for p, dpr_m in zip(batch["prompt"], self._dpr_sampling_mask) if dpr_m]
        )

        # Conditions for generating a batch of samples
        if (
            (
                # If insufficient outputs for DPP
                self.p > 0
                and (
                    len(self._dpp_generation_outputs)
                    < self.args.per_device_train_batch_size
                )
            )
            or (
                # If insufficient outputs for DPR
                self.g > 0
                and (
                    len(self._dpr_generation_outputs)
                    < self.args.per_device_train_batch_size
                )
            )
        ) and (
            # If sufficient inputs collected
            len(self._dpp_generation_inputs)
            + len(self._dpr_generation_inputs)
            + self.args.per_device_train_batch_size
            > self.per_device_generation_batch_size * self.generation_num_batches
        ):
            self._generate_batch_samples(model)

        # Return the updated batch
        return batch

    ##############################

    ##############################
    # DPP & DPR
    def _generate_batch_samples(self, model):

        # Empty CUDA memory cache before generating samples
        torch.cuda.empty_cache()

        # Shuffle and truncate the collected prompts
        prompt_list = np.array(
            self._dpp_generation_inputs + self._dpr_generation_inputs
        )
        prompt_origin_mask = np.array(
            [True] * len(self._dpp_generation_inputs)
            + [False] * len(self._dpr_generation_inputs)
        )
        prompt_shuffle_indices = np.random.permutation(len(prompt_list))
        prompt_list = prompt_list[prompt_shuffle_indices][
            : self.per_device_generation_batch_size * self.generation_num_batches
        ]
        prompt_origin_mask = prompt_origin_mask[prompt_shuffle_indices][
            : self.per_device_generation_batch_size * self.generation_num_batches
        ]

        # Tokenize the collected prompts
        prompt_tokens = [
            self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_prompt_length,
                add_special_tokens=True,
            )
            for prompt in prompt_list
        ]

        prompt_tokens_batches = [
            prompt_tokens[
                i : min(i + self.per_device_generation_batch_size, len(prompt_tokens))
            ]
            for i in range(0, len(prompt_tokens), self.per_device_generation_batch_size)
        ]
        prompt_batches = [
            self._prepare_inputs(
                self.data_collator(
                    [
                        {
                            "prompt_input_ids": p["input_ids"],
                            "prompt_attention_mask": p["attention_mask"],
                        }
                        for p in pt
                    ]
                )
            )
            for pt in prompt_tokens_batches
        ]

        # If one uses `generate_during_eval` with peft + bf16, we need to explictly call generate with
        # the torch cuda amp context manager as some hidden states are silently casted to full precision.
        generate_context_manager = (
            nullcontext
            if not self._peft_has_been_casted_to_bf16
            else torch.cuda.amp.autocast
        )

        # Generate a batch of responses from the model
        generated_text = []
        with generate_context_manager():
            with unwrap_model_for_generation(
                model, self.accelerator
            ) as unwrapped_model:
                with torch.no_grad():
                    for prompt_batch in prompt_batches:
                        generated_ids_batch = unwrapped_model.generate(
                            input_ids=prompt_batch["prompt_input_ids"],
                            attention_mask=prompt_batch["prompt_attention_mask"],
                            max_length=None,
                            max_new_tokens=self.max_target_length,
                            do_sample=True,
                            temperature=self.generation_temperature,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            # Generate two responses for each prompt
                            num_return_sequences=2,
                        )

                        # Decode the generated responses
                        generated_text_batch = self.tokenizer.batch_decode(
                            pad_to_length(
                                generated_ids_batch,
                                self.max_length,
                                self.tokenizer.pad_token_id,
                            ),
                            skip_special_tokens=True,
                        )
                        generated_text.extend(generated_text_batch)

        # Pack the generated responses into features
        generated_features = [
            {
                "prompt": prompt,
                "chosen": chosen[len(prompt) :],
                "rejected": rejected[len(prompt) :],
            }
            for prompt, chosen, rejected in zip(
                (self._dpp_generation_inputs + self._dpr_generation_inputs),
                # Unroll the generated responses
                generated_text[::2],
                generated_text[1::2],
            )
        ]
        assert len(generated_features) == len(prompt_list)

        # Set the generation outputs
        self._dpp_generation_outputs = self._label_generated_by_polcy(
            model, [f for f, m in zip(generated_features, prompt_origin_mask) if m]
        )

        self._dpr_generation_outputs = self._label_generated_by_reward(
            [f for f, m in zip(generated_features, prompt_origin_mask) if not m]
        )

        # Duplicate the generated outputs and shuffle
        self._dpp_generation_outputs = [
            f
            for f in self._dpp_generation_outputs
            for _ in range(self.generation_reuse_multiplier)
        ]
        self._dpr_generation_outputs = [
            f
            for f in self._dpr_generation_outputs
            for _ in range(self.generation_reuse_multiplier)
        ]
        random.shuffle(self._dpp_generation_outputs)
        random.shuffle(self._dpr_generation_outputs)

        # Clear generation input buffers
        self._dpp_generation_inputs = []
        self._dpr_generation_inputs = []

        # Empty CUDA memory cache after generating samples
        torch.cuda.empty_cache()

    ##############################

    ##############################
    # DPP
    def _label_generated_by_polcy(
        self, model, generated_features: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:

        # If no samples are to be labeled, return the generated features as is
        if not generated_features:
            return generated_features

        # Empty CUDA memory cache before labeling samples
        torch.cuda.empty_cache()

        # Split the generated features into batches according to the training batch size
        logits = []
        for batch_features in [
            generated_features[i : i + self.args.per_device_train_batch_size]
            for i in range(
                0, len(generated_features), self.args.per_device_train_batch_size
            )
        ]:
            # Prepare the batch for the model
            generated_batch = self.data_collator(
                [self.tokenize_row(f, model) for f in batch_features]
            )
            generated_batch = self._prepare_inputs(generated_batch)

            # Calculate the log probabilities of the chosen (first) and rejected (second) responses
            with torch.no_grad():
                # Stop gradients from flowing back to the model
                (
                    policy_chosen_logps,
                    policy_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_forward(model, generated_batch)
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, generated_batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, generated_batch)
                pi_logratios = policy_chosen_logps - policy_rejected_logps
                ref_logratios = reference_chosen_logps - reference_rejected_logps
                logits.append((pi_logratios - ref_logratios).cpu())

        logits = torch.cat(logits, dim=0)

        # Sampling whether switching the chosen (first) and rejected (second) responses
        switching_mask = torch.bernoulli(1 - F.sigmoid(self.beta * logits)).bool()

        # Switch the chosen and rejected responses according to the switching mask
        for f, m in zip(generated_features, switching_mask):
            if m:
                f["chosen"], f["rejected"] = f["rejected"], f["chosen"]

        return generated_features

    ##############################

    ##############################
    # DPR
    def _label_generated_by_reward(
        self, generated_features: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:

        # If no samples are to be labeled, return the generated features as is
        if not generated_features:
            return generated_features

        # Empty CUDA memory cache before labeling samples
        torch.cuda.empty_cache()

        # Split the generated features into batches according to the training batch size
        chosen_scores, rejected_scores = [], []
        for batch_features in [
            generated_features[i : i + self.per_device_evalreward_batch_size]
            for i in range(
                0, len(generated_features), self.per_device_evalreward_batch_size
            )
        ]:
            # Load model to GPU
            self.reward_model = self.reward_model.to(self.accelerator.device)
            with torch.no_grad():
                chosen_output = self.reward_model(
                    **self.reward_tokenizer(
                        [f["prompt"] + f["chosen"] for f in batch_features],
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.reward_model.device)
                )
                rejected_output = self.reward_model(
                    **self.reward_tokenizer(
                        [f["prompt"] + f["rejected"] for f in batch_features],
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.reward_model.device)
                )
                if self.reward_model_id.startswith("PKU-Alignment"):
                    chosen_output, rejected_output = (
                        chosen_output.end_scores.cpu().flatten(),
                        rejected_output.end_scores.cpu().flatten(),
                    )
                elif self.reward_model_id.startswith("openbmb"):
                    chosen_output, rejected_output = (
                        chosen_output.cpu().flatten(),
                        rejected_output.cpu().flatten(),
                    )
                else:
                    raise RuntimeError("Unknown reward model")
                chosen_scores.append(chosen_output)
                rejected_scores.append(rejected_output)
            # Offload model from GPU
            self.reward_model = self.reward_model.cpu()

        chosen_scores = torch.cat(chosen_scores, dim=0)
        rejected_scores = torch.cat(rejected_scores, dim=0)

        # Switch the chosen and rejected responses according to the switching mask
        for f, cs, rs in zip(generated_features, chosen_scores, rejected_scores):
            if ((not self.reward_model_reverse) and (cs < rs)) or (
                self.reward_model_reverse and (cs > rs)
            ):
                f["chosen"], f["rejected"] = f["rejected"], f["chosen"]

        return generated_features

    ##############################

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = (
            torch.cuda.amp.autocast
            if self._peft_has_been_casted_to_bf16
            else nullcontext
        )

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(
                model, inputs, train_eval="eval"
            )

        # force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(
            v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys
        )
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(
        self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval:
            # Generate random indices within the range of the total number of samples
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(
                range(num_samples), k=self.args.eval_batch_size
            )

            # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded, ref_output_decoded = self.get_batch_samples(
                self.model, random_batch
            )

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy", "Ref Model"],
                        rows=[
                            [prompt, pol[len(prompt) :], ref[len(prompt) :]]
                            for prompt, pol, ref in zip(
                                random_batch["prompt"],
                                policy_output_decoded,
                                ref_output_decoded,
                            )
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        # Base evaluation
        initial_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(
        self,
        revision=None,
        commit_message: Optional[str] = "End of training",
        blocking: bool = True,
        **kwargs,
    ) -> str:
        """
        Override Trainer.push_to_hub to allow push to hub to specific revision.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(
            model=self.model, tag_names=self._tag_names, kwargs=kwargs
        )

        model_name = kwargs.pop("model_name", None)
        if model_name is None and self.args.should_save:
            if self.args.hub_model_id is None:
                model_name = Path(self.args.output_dir).name
            else:
                model_name = self.args.hub_model_id.split("/")[-1]

        # In case the user calls this method with args.push_to_hub = False
        if self.hub_model_id is None:
            self.init_hf_repo()

        # Create a new branch if revision is provided
        if revision is not None:
            create_branch(
                repo_id=self.hub_model_id,
                branch=revision,
                token=self.args.hub_token,
                exist_ok=True,
            )

        # Needs to be executed on all processes for TPU training, but will only save on the processed determined by
        # self.args.should_save.
        self.save_model(_internal_call=True)

        # Only push from one node.
        if not self.is_world_process_zero():
            return

        # Add additional tags in the case the model has already some tags and users pass
        # "tags" argument to `push_to_hub` so that trainer automatically handles internal tags
        # from all models since Trainer does not call `model.push_to_hub`.
        if getattr(self.model, "model_tags", None) is not None:
            if "tags" not in kwargs:
                kwargs["tags"] = []

            # If it is a string, convert it to a list
            if isinstance(kwargs["tags"], str):
                kwargs["tags"] = [kwargs["tags"]]

            for model_tag in self.model.model_tags:
                if model_tag not in kwargs["tags"]:
                    kwargs["tags"].append(model_tag)

        self.create_model_card(model_name=model_name, **kwargs)

        # Wait for the current upload to be finished.
        self._finish_current_push()
        return upload_folder(
            repo_id=self.hub_model_id,
            revision=revision,
            folder_path=self.args.output_dir,
            commit_message=commit_message,
            token=self.args.hub_token,
            run_as_future=not blocking,
            ignore_patterns=["_*", "checkpoint-*"],
        )

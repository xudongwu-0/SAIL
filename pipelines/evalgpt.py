import os
import sys
import re
from dataclasses import dataclass, field
import itertools
from tqdm.auto import tqdm
from tqdm.asyncio import tqdm_asyncio
from huggingface_hub import HfApi
import tiktoken
from openai import AsyncAzureOpenAI, AsyncOpenAI, RateLimitError
from aiolimiter import AsyncLimiter
import asyncio
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import HfArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import CONFIGS, sample_every_k_batched


HUGGINGFACE_CONFIGS = CONFIGS.services.huggingface
HFAPI = HfApi(HUGGINGFACE_CONFIGS["token"])
GPT_CONFIGS = CONFIGS.evaluations.gpt
OPENAI_CONFIGS = CONFIGS.services.openai
CACHE_CONFIGS = CONFIGS.utils.cache
TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass
class ScriptArguments:
    run_name: str = field(
        metadata={"help": "run name to evaluate"},
    )
    tag: str = field(
        metadata={"help": "tag for the experiment"},
    )
    gpt_ver: str = field(
        default="gpt-35-turbo",
        metadata={"help": "version of GPT to evaluate"},
    )
    every_k: int = field(
        default=1,
        metadata={
            "help": "evaluate every k samples, if a fraction, evaluate each sample 1/k times"
        },
    )
    batch_size: int = field(
        default=256,
        metadata={"help": "batch size for parallel calling chat api"},
    )
    max_retries: int = field(
        default=3,
        metadata={"help": "max retries for calling chat api"},
    )
    max_tokens: int = field(
        default=1000,
        metadata={"help": "max tokens requested for each completion"},
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "temperature for sampling"},
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "top p for sampling"},
    )
    frequency_penalty: float = field(
        default=0,
        metadata={"help": "frequency penalty for sampling"},
    )
    presence_penalty: float = field(
        default=0,
        metadata={"help": "presence penalty for sampling"},
    )
    is_pairwise: bool = field(
        default=True,
        metadata={"help": "whether the prompt is pairwise comparison"},
    )
    system_prompt: str = field(
        default="none",
        metadata={"help": "system prompt for GPT, do not change"},
    )
    user_prompt: str = field(
        default="none",
        metadata={"help": "user prompt for GPT, do not change"},
    )
    prompt_tokens: int = field(
        default=-1,
        metadata={"help": "estimated tokens per prompt, do not change"},
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


# Call openai api in async manner with rate limit, support criteria and retries
async def call_openai_api(
    request, criteria, max_retries, client, model_name, rate_limiter
):
    request["model"] = model_name
    counter = 0
    while counter < max_retries:
        async with rate_limiter:
            try:
                result = await asyncio.wait_for(
                    client.chat.completions.create(**request), timeout=60
                )
                content = result.choices[0].message.content
                if isinstance(content, str) and criteria(content):
                    return content
            except asyncio.TimeoutError:
                print(f"Request timed out")
                counter += 1
            except RateLimitError:
                await asyncio.sleep(5)
                counter += 1
            except Exception as e:
                print(f"Error: {e}")
                counter += 1
    return None


async def iterate_call_openai_api(
    request, criteria, max_retries, client, model_name, rate_limiter
):
    content = []
    for message in request["messages"]:
        single_request = request.copy()
        single_request["messages"] = message
        content.append(
            await call_openai_api(
                request=single_request,
                criteria=criteria,
                max_retries=max_retries,
                client=client,
                model_name=model_name,
                rate_limiter=rate_limiter,
            )
        )
    return content


# Async distribute multiple api calls to multiple clients
async def round_robin_calls(requests, endpoints, criteria, max_retries):
    tasks = [
        iterate_call_openai_api(
            request=request, criteria=criteria, max_retries=max_retries, **endpoint
        )
        for request, endpoint in zip(requests.values(), itertools.cycle(endpoints))
    ]
    contents = await tqdm_asyncio.gather(*tasks)
    return {idx: content for idx, content in zip(requests.keys(), contents)}


# Evaluate reward of responses
async def evaluate_gpt(response_dataset, script_args):
    generator, num_iters = sample_every_k_batched(
        response_dataset, script_args.every_k, batch_size=script_args.batch_size
    )
    endpoints = [
        {
            "client": (
                AsyncAzureOpenAI(
                    azure_endpoint=client_config["azure_endpoint"],
                    api_key=client_config["api_key"],
                    api_version=client_config["api_version"],
                )
                if "azure_endpoint" in client_config
                else AsyncOpenAI(
                    base_url=client_config["base_url"],
                    api_key=client_config["api_key"],
                )
            ),
            "model_name": client_config["model_name"],
            "rate_limiter": AsyncLimiter(
                min(
                    client_config["RPM"],
                    client_config["TPM"]
                    // (script_args.prompt_tokens + script_args.max_tokens),
                ),
                60,
            ),
        }
        for client_config in OPENAI_CONFIGS[script_args.gpt_ver]
    ]

    results = {}
    for indices, samples in tqdm(generator, total=num_iters):
        # Populating api requests
        requests = {}
        switch_orders = {}
        for idx, sample in zip(indices, samples):
            # Randomly switch the order of the responses with 50% probability
            switch_orders[idx] = np.random.rand() > 0.5
            if script_args.is_pairwise:
                messages = [
                    [
                        {"role": "system", "content": script_args.system_prompt},
                        {
                            "role": "user",
                            "content": script_args.user_prompt.format(
                                prompt=sample["prompt"],
                                # Default to the first response as chosen and the second as the model's response
                                answer1=(
                                    sample["chosen"]
                                    if not switch_orders[idx]
                                    else sample["response"]
                                ),
                                answer2=(
                                    sample["response"]
                                    if not switch_orders[idx]
                                    else sample["chosen"]
                                ),
                            ),
                        },
                    ]
                ]
            else:
                messages = [
                    [
                        {"role": "system", "content": script_args.system_prompt},
                        {
                            "role": "user",
                            "content": script_args.user_prompt.format(
                                prompt=sample["prompt"],
                                answer=(
                                    sample["chosen"]
                                    if not switch_orders[idx]
                                    else sample["response"]
                                ),
                            ),
                        },
                    ],
                    [
                        {"role": "system", "content": script_args.system_prompt},
                        {
                            "role": "user",
                            "content": script_args.user_prompt.format(
                                prompt=sample["prompt"],
                                answer=(
                                    sample["response"]
                                    if not switch_orders[idx]
                                    else sample["chosen"]
                                ),
                            ),
                        },
                    ],
                ]
            requests[idx] = {
                # Leave the model field as None for the chat api to choose the model
                "model": None,
                "messages": messages,
                "max_tokens": script_args.max_tokens,
                "temperature": script_args.temperature,
                "top_p": script_args.top_p,
                "frequency_penalty": script_args.frequency_penalty,
                "presence_penalty": script_args.presence_penalty,
                "stop": None,
            }

        # Call the chat api and ensure the criteria
        contents = await round_robin_calls(
            requests=requests,
            endpoints=endpoints,
            criteria=lambda content: any(
                bool(re.search(pattern, content))
                for pattern in script_args.match_patterns
            ),
            max_retries=script_args.max_retries,
        )
        # Collecting results
        for idx, content in contents.items():
            # Skip if the content is None
            if content[0] is None or (
                not script_args.is_pairwise and content[1] is None
            ):
                continue
            if idx not in results:
                results[idx] = []
            # Parse the scores and explanation from the chat api response
            # Because of ensured criteria, this will always succeed
            for pattern in script_args.match_patterns:
                if re.search(pattern, content[0]):
                    if script_args.is_pairwise:
                        score1, score2 = tuple(
                            map(float, re.findall(pattern, content[0])[-1])
                        )
                    else:
                        score1 = float(re.findall(pattern, content[0])[-1])
                        score2 = float(re.findall(pattern, content[1])[-1])
                        break
            # Reverse the scores if the order was switched
            results[idx].append(
                (score2 - score1) if not switch_orders[idx] else (score1 - score2)
            )

    # Add or replace the 'gpt_score' column in the dataset
    response_dataset = (
        response_dataset.remove_columns("gpt_score")
        if "gpt_score" in response_dataset.column_names
        else response_dataset
    )
    response_dataset = response_dataset.add_column(
        "gpt_score",
        [
            np.mean(results[idx]) if idx in results else None
            for idx in range(len(response_dataset))
        ],
    )
    return response_dataset


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Dataset
    response_dataset = load_generated_dataset(script_args)

    # Find the GPT template by dataset type
    for dataset_prefix in GPT_CONFIGS.keys():
        if script_args.run_name.split("_")[2].startswith(dataset_prefix):
            script_args.is_pairwise = GPT_CONFIGS[dataset_prefix]["is_pairwise"]
            script_args.system_prompt = GPT_CONFIGS[dataset_prefix]["system_prompt"]
            script_args.user_prompt = GPT_CONFIGS[dataset_prefix]["user_prompt"]
            script_args.match_patterns = GPT_CONFIGS[dataset_prefix]["match_patterns"]
            script_args.prompt_tokens = (
                len(
                    TOKENIZER.encode(
                        script_args.system_prompt
                        + script_args.user_prompt
                        + response_dataset[0]["prompt"]
                        + response_dataset[0]["chosen"]
                        + response_dataset[0]["response"]
                    )
                )
                + 100
            )
            break

    # Evaluation
    response_dataset = asyncio.run(evaluate_gpt(response_dataset, script_args))

    # Push to Hub
    DatasetDict(
        {"default": response_dataset},
    ).push_to_hub(
        HUGGINGFACE_CONFIGS["prefix"]["evaluations"] + script_args.run_name,
        script_args.tag,
    )


if __name__ == "__main__":
    main()

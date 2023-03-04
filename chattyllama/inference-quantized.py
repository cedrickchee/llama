# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

# A copy of `example.py` but with our changes.
# Basically, we removed all Model Parallel (MP) stuffs, so we can run 7B/13B
# model in a single GPU.
#
# Modified by: https://github.com/cedrickchee/llama

# from typing import Tuple
# import os
# import sys
import torch
# import fire
import time
import json

from pathlib import Path

# from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


# def setup_model_parallel() -> Tuple[int, int]:
#     local_rank = int(os.environ.get("LOCAL_RANK", -1))
#     world_size = int(os.environ.get("WORLD_SIZE", -1))

#     torch.distributed.init_process_group("nccl")
#     initialize_model_parallel(world_size)
#     torch.cuda.set_device(local_rank)

#     # seed must be the same in all processes
#     torch.manual_seed(1)
#     return local_rank, world_size

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

    # removed world_size and local_rank.

    # assert world_size == len(
    #     checkpoints
    # ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}" 
    
    # ckpt_path = checkpoints[local_rank]
    # print("Loading")
    # checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    
    key_to_dim = {
        "w1": 0,
        "w2": -1,
        "w3": 0,
        "wo": -1,
        "wq": 0,
        "wk": 0,
        "wv": 0,
        "output": 0,
        "tok_embeddings": -1,
        "ffn_norm": None,
        "attention_norm": None,
        "norm": None,
        "rope": None,
    }
    print("Loading")
    for i, ckpt in enumerate(checkpoints):
        checkpoint = torch.load(ckpt, map_location="cpu")
        for parameter_name, parameter in model.named_parameters():
            short_name = parameter_name.split(".")[-2]
            if key_to_dim[short_name] is None and i == 0:
                parameter.data = checkpoint[parameter_name]
            elif key_to_dim[short_name] == 0:
                size = checkpoint[parameter_name].size(0)
                parameter.data[size * i : size * (i + 1), :] = checkpoint[
                    parameter_name
                ]
            elif key_to_dim[short_name] == -1:
                size = checkpoint[parameter_name].size(-1)
                parameter.data[:, size * i : size * (i + 1)] = checkpoint[
                    parameter_name
                ]
        del checkpoint
    
    # bnb quantize the model!
    model.quantize()

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    # local_rank, world_size = setup_model_parallel()
    # if local_rank > 0:
    #     sys.stdout = open(os.devnull, "w")
    # local_rank = 0
    # world_size = 1

    generator = load(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)    

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is", # removed: keep only one prompt
    ]
    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    # fire.Fire(main)
    
    # hard-coded params for easier testing.
    main(
        ckpt_dir="13B",
        tokenizer_path="tokenizer.model",
        temperature=0.8,
        top_p=0.95,
        max_seq_len=1024,
        max_batch_size=1
    )

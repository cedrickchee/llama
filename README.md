# LLaMA

This is a variant of the LLaMA model and has the following changes:
- **Compression:** 8-bit model quantization using [bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration)
- **Non-Model Parallel (MP):** run 13B model in a single GPU. All MP codes removed.
- **Extended model:**
  - Fix the sampler — a better sampler that improve generations quality: `temperature`, `top_p`, `repetition_penalty`, `tail_free`.
  - (Future): provides more controls for generations, expose repetition penalty so that CLI can pass-in the options.

And more soon. I'm [experimenting with compression and acceleration techniques](./chattyllama/README.md#llama-model-weights) to make the models:
- smaller and faster
- run on low-resources hardwares

I'm also building LLaMA-based ChatGPT.

## Hardware

- [Guide on GPU requirements](./chattyllama/hardware.md#guide-on-gpu-requirements)
- [Memory requirements for each model size](./chattyllama/hardware.md#memory-requirements-for-each-model-size)

## ChattyLLaMA

[ChattyLLaMA](./chattyllama/) is **experimental** LLaMA-based ChatGPT.

### Documentations

All the new codes are available in the [chattyllama](./chattyllama/) directory.

**Combined**

All changes and fixes baked into one:
- Non-Model Parallel (MP): all MP constructs removed (MP shards weights across a GPU cluster setup)
- 8-bit quantized model using bitsandbytes
- Sampler fixes, better sampler

Source files location:
- `chattyllama/combined/model.py`: a fork of LLaMA model.
- `chattyllama/combined/inference.py`: run model inference (it's a modified copy of `example.py`).

**Non-MP/single GPU**

Source files location:
- `chattyllama/model.py`: a fork of LLaMA model.
- `chattyllama/inference.py`: run model inference

### Code Examples

Code walkthrough: [notebooks](./notebooks/).

This shows how you can get it running on 1x A100 40GB GPU. The code is outdated though. It's using the original model version from MetaAI.

For bleeding edge things, follow the below quick start.

#### Quick start

1. Download model weights into `./model`.

2. Install all the needed dependencies.

```sh
$ git clone https://github.com/cedrickchee/llama.git
$ cd llama && pip install -r requirements.txt
```

**Note:**

- Don't use Conda. Use pip.
- If you have trouble with bitsandbytes, [build and install it from source](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md).

```sh
$ pip install -e .
#torchrun --nproc_per_node 1 example.py --ckpt_dir ../7B --tokenizer_path ../tokenizer.model
$ cd chattyllama/combined
```

3. Modify `inference.py` with the path to your weights directory:

```py
# ...

if __name__ == "__main__":
    main(
        ckpt_dir="/model/vi/13B", # <-- change the path
        tokenizer_path="/model/vi/tokenizer.model", # <-- change the path
        temperature=0.7,
        top_p=0.85,
        max_seq_len=1024,
        max_batch_size=1
    )
```

4. Modify `inference.py` with your prompt:

```py
def main(...):
    # ...

    prompts = [
        "I believe the meaning of life is"
    ]

    # ...
```

5. Run inference:

```sh
$ python inference.py
```

### LLaMA compatible port

Looking to use LLaMA model with HuggingFace library?
Well look at [my "transformers-llama" repo](https://github.com/cedrickchee/transformers-llama).

#### Other ports

- [Text generation web UI](https://github.com/oobabooga/text-generation-webui) - A Gradio Web UI for running Large Language Models like LLaMA, GPT-Neo, OPT, and friends. My guide: ["Installing 8/4-bit LLaMA with text-generation-webui on Linux"](https://gist.github.com/cedrickchee/1f24fa3a5e3371910e1959b96a8dff94)
- [LLaMa CPU fork](https://github.com/markasoftware/llama-cpu) - We need more work like this that lower the compute requirements. Really under appreciated.
- [LLaMA Jax](https://github.com/Sea-Snell/JAX_llama)
- [Minimal LLaMA](https://github.com/cedrickchee/minimal-llama) - Jason's HuggingFace Transformers port using OPT code internally. This version should be more stable. But the code is not well-tested yet. Bonus: you can quickly see how well the model can be fine-tuned either using HuggingFace PEFT with 8-bit or Pipeline Parallelism.
- [Running LLaMA 7B on a 64GB M2 MacBook Pro with llama.cpp](https://til.simonwillison.net/llms/llama-7b-m2) by Simon Willison - llama.cpp is from the same Whisper.cpp hacker, ggerganov. Never dissapointed by ggerganov's work.
  > It's genuinely possible to run a LLM that's hinting towards the performance of GPT3 on your own hardware now. I thought that was still a few years away.

   Looking at this rate of model compression/acceleration progress, soon we can run a LLM inference locally on mobile devices. QNNPACK, a hardware optimized library that also supports mobile processors can help. JIT compiler like OpenXLA/PyTorch Glow can optimize the computation graph so the model runs well on low-resources hardware.
  
  We underestimated pre-trained language models (~2019) and overestimated a lot of things.

  My [llama.cpp](https://github.com/cedrickchee/llama.cpp) patches for Linux support. (WIP)

<details>
<summary>See more</summary>

- [pyllama](https://github.com/juncongmoo/pyllama) - Run LLM in a single GPU, as simple as `pip install pyllama`. It's a quick & dirty hacked version of 🦙 LLaMA. Bonus: comes with a way to start a Gradio Web UI for trying out prompting in browser. Good tips: "To load KV cache in CPU, run `export KV_CAHCHE_IN_GPU=0` in the shell.".
- [minichatgpt](https://github.com/juncongmoo/minichatgpt) - Train ChatGPT in minutes with [ColossalAI (blog post)](https://www.hpc-ai.tech/blog/colossal-ai-chatgpt) (minichatgpt training process is pending my verification. I can confirm the code there was based on ColossalAI's [mini demo](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ChatGPT). It doesn't support LLaMA yet.)
  - Supports LoRA
  - Supports RL paradigms, like reward model, PPO
  - Datasets used for training:
    - Train with prompt data from: [fka/awesome-minichatgpt-prompts](https://huggingface.co/datasets/fka/awesome-minichatgpt-prompts). Training scripts and instructions [here](https://github.com/juncongmoo/minichatgpt/tree/main/examples#train-with-real-prompt-data).
    - Train the reward model using [Dahoas/rm-static](https://huggingface.co/datasets/Dahoas/rm-static) dataset.
</details>

### Supporting tools

- [Resharding and HuggingFace conversion](https://github.com/dmahan93/llama/blob/main/CONVERSIONS.md) - Useful scripts for transforming the weights, if you still want to spread the weights and run the larger model (in fp16 instead of int8) across multiple GPUs for some reasons.

### Plan

TODO:

**Priority: high**

- [ ] Improve sampler - refer to [shawwn/llama](https://github.com/shawwn/llama) fork.
- [ ] Fine-tune the models on a diverse set of instructions datasets from LAION's
OpenAssistant. Check out my [ChatGPT notes](https://github.com/cedrickchee/chatgpt-universe#training-data) for larger training data. (blocked by dataset v1)
- [ ] Try the fine-tuning protocol from Flan.
  - LLaMA paper touches on finetuning briefly, referencing that.
- [ ] Fine-tune model with HF's PEFT and Accelerate. PEFT doesn't support causal LM like LLaMA yet (blocked by [PR](https://github.com/huggingface/peft/pull/160))

**Priority: low**

- [ ] Start and try other fine-tuning ideas:
  - ChatGPT-like = LLaMA + CarperAI's tRLX (RLHF) library + Anthropic's public preference dataset. I don't know how feasible if the experiments are larger scale (compute-wise) that use RL models that are good at instruction following.

Reminder-to-self:

- People under-appreciate fine-tuning alone compared to RLHF. RL algorithms
(unsupervised) are quite finicky compared to supervised deep learning. RL is
hard-ish.

---

# Original README

This repository is intended as a minimal, hackable and readable example to load [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models and run inference.
In order to download the checkpoints and tokenizer, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5)

## Setup

In a conda env with pytorch / cuda available, run:
```
pip install -r requirements.txt
```
Then in this repository:
```
pip install -e .
```

## Download

Once your request is approved, you will receive links to download the tokenizer and model files.
Edit the `download.sh` script with the signed url provided in the email to download the model weights and tokenizer.

## Inference

The provided `example.py` can be run on a single or multi-gpu node with `torchrun` and will output completions for two pre-defined prompts. Using `TARGET_FOLDER` as defined in `download.sh`:
```
torchrun --nproc_per_node MP example.py --ckpt_dir $TARGET_FOLDER/model_size --tokenizer_path $TARGET_FOLDER/tokenizer.model
```

Different models require different MP values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 33B    | 4  |
| 65B    | 8  |

## FAQ

- [1. The download.sh script doesn't work on default bash in MacOS X](FAQ.md#1)
- [2. Generations are bad!](FAQ.md#2)
- [3. CUDA Out of memory errors](FAQ.md#3)
- [4. Other languages](FAQ.md#4)

## Reference

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

```
@article{touvron2023llama,
  title={LLaMA: Open and Efficient Foundation Language Models},
  author={Touvron, Hugo and Lavril, Thibaut and Izacard, Gautier and Martinet, Xavier and Lachaux, Marie-Anne and Lacroix, Timoth{\'e}e and Rozi{\`e}re, Baptiste and Goyal, Naman and Hambro, Eric and Azhar, Faisal and Rodriguez, Aurelien and Joulin, Armand and Grave, Edouard and Lample, Guillaume},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```

## Model Card
See [MODEL_CARD.md](MODEL_CARD.md)

## License
See the [LICENSE](LICENSE) file.

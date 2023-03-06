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

---

# Original README

This repository is intended as a minimal, hackable and readable example to load [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) ([arXiv](https://arxiv.org/abs/2302.13971v1)) models and run inference.
In order to download the checkpoints and tokenizer, fill this [google form](https://forms.gle/jk851eBVbX1m5TAv5)

## Setup

In a conda env with pytorch / cuda available, run
```
pip install -r requirements.txt
```
Then in this repository
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

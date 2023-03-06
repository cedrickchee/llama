# Hardware infrastructure

All my model runs and inference results are captured in my [notebooks](../notebooks/).

You can see the GPU requirements (NVIDIA card name, VRAM size) for my experiments there.

Below is a collection of model and hardware test results, including mine.

## Guide on GPU requirements

How much GPU VRAM do I need to run the 7B model? Can 65B fit on 4x RTX 3090? Out of memory?

Here's a table showing a working combination of model sizes, GPU name, and VRAM
requirements. Some data points are reported by other users.

| Model  | GPU        | Min. VRAM  | VRAM Used | Precision | Environment   | Throughput |
|--------|------------|------------|-----------|-----------|---------------|------------|
| 7B     | A100 40GB  | 40GB       | No OOM    | TF32      | Colab Pro     | <1 tok/s   |
| 7B^    | 3090 24GB  | 24GB       | No OOM    | FP32      | Home PC       | ??         |
| 7B     | A4000 16GB | 16GB       | No OOM    | TF32      | Home PC       | ??         |
| 7B**   | 3060 12GB  | 10GB       | 9.1GB     | int8      | Unspecified   | 4-9it/s    |
| 7B**   | 3080 10GB  | 10GB       | 9.2GB     | int8      | Unspecified   | ??         |
| 7B**   | 3090 24GB  | 10GB       | 9.4GB     | int8      | Unspecified   | 29-55it/s  |
| 7B^^   | 2080 8GB   | 8GB        |           | FP32      | Home PC       | ??         |

| Model  | GPU        | Min. VRAM  | VRAM Used | Precision | Environment   | Throughput |
|--------|------------|------------|-----------|-----------|---------------|------------|
| 13B**  | 3090Ti 24GB| 20GB       | 16.2GB     | int8     | Unspecified   | 13-29 it/s |
| 13B**  | 4090 24GB  | 20GB       | 16.5GB     | int8     | Unspecified   | 11-32 it/s |

| Model  | GPU        | Min. VRAM  | VRAM Used | Precision | Environment   | Throughput |
|--------|------------|------------|-----------|-----------|---------------|------------|
| 33B**  | A6000 48GB | 40GB       | 35.8GB    | int8      | Unspecified   | 19-38 it/s |
| 33B**  | A100 40GB  | 40GB       | 36.2GB    | int8      | Unspecified   | 21-39 it/s |

| Model  | GPU        | Min. VRAM  | VRAM Used | Precision | Environment   | Throughput |
|--------|------------|------------|-----------|-----------|---------------|------------|
| 65B**  | A100 80GB  | 80GB       | ~74.3GB   | int8      | Unspecified   | 15-35 it/s |

(_WIP: incomplete, accuracy is not fully confirm yet._)

_^ A modified of LLaMA model (`model.py`), configured for running with a single GPU (default is distributed GPU).Lowered batch size to 1 so the model can fit within VRAM._

_^^ A modified of LLaMA model. Only keep a single transformer block on the GPU at a time. Changed from fairscale layers to `torch.nn.Linear`. Details, see [this GitHub Issue](https://github.com/facebookresearch/llama/issues/79#issuecomment-1454707663)._

_** 8-bit quantized model._

### Memory requirements for each model size

Model arguments:
- `max_batch_size`: 1 (**IMPORTANT**)
- `max_seq_length`: 1024

| Model Params (Billions)  | 6.7  | 13   | 32.5 | 65.2 |
| ------------------------ | ---- | ---- | ---- | ---- |
| n_layers                 | 32   | 40   | 60   | 80   |
| n_heads                  | 32   | 40   | 52   | 64   |
| dim                      | 4096 | 5120 | 6656 | 8192 |

The above numbers are gathered from the paper (Table 2).

Memory requirements in fp16 precision (before int8 quantization):

| Model Params (Billions)  | 6.7  | 13   | 32.5 | 65.2 |
|--------------------------|------|------|------|------|
| Model on disk (GB)***    | 13   | 26   | 65   | 130  |
| Cache (GB)               | 1    | 1    | 2    | 3    |
| Total (GB)               | 14   | 27   | 67   | 133  |

Transformer kv cache (decoding cache) formula:
- Per token (bytes) = `cache_per_token = 2 * 2 * n_layers * n_heads * head_dim` [^3]
  
  Example for 7B: `cache_per_token = 2 * 2 * 32 * 32 * (4096 / 32) = 524288`. `head_dim = dim / n_heads`
- Total (bytes) = `total = cache_per_token * max_batch_size * max_seq_len`
  
  Example for 7B: `total = 524288 * 1 * 1024 = 536870912` (~1GB)

**Example:** 7B require at least 14GB in 16-bit (fp16) precision or 7GB VRAM in 8-bit (int8) precision (half of VRAM).

(_[LLaMA's FAQ about this](https://github.com/facebookresearch/llama/blob/main/FAQ.md#3-cuda-out-of-memory-errors) is confusing. I calculate based on the spreadsheet below (not mine). The previous formula here was based on it._)


_*** Model (on disk) is the total file size of `consolidated.XX.pth` for a model. For example, 13B is 24GB because it has two `consolidated.XX.pth` files, each has a file size of 12GB. Weight file sizes are below._

![llama-model-weights-resized](https://user-images.githubusercontent.com/145605/222949548-4970b717-64e4-482f-b9d2-ab77669b11cb.png)

Well, 65B needs a GPU cluster with a total of 125GB VRAM in int8 precison or 250GB in fp16.

[Spreadsheet](https://docs.google.com/spreadsheets/d/1EsRZcsvbITBCfb5N1toir5huomZxIq4lDNwQW3NfPRE/edit)
to calculate the memory requirements for each model size, following the FAQ and
paper. You can make a copy to adjust the batch size and sequence length.[^1]

Some people just made enough code changes to **run the 7B model on the CPU**. [^2]
I can't confirm this though.

### Troubleshooting

- I experienced an out-of-memory (OOM) error or something along the line.

  I don't provide tech support. I suggest you ask for help at: https://github.com/oobabooga/text-generation-webui/issues/147

[^1]: [GitHub Issue: Post your hardware specs here if you got it to work](https://github.com/facebookresearch/llama/issues/79#issuecomment-1453814121)
[^2]: [GitHub Issue reply by "gmorenz"](https://github.com/facebookresearch/llama/issues/79#issuecomment-1454042028)
[^3]: [Transformer Inference Arithmetic](https://kipp.ly/blog/transformer-inference-arithmetic/#capacity)
# Hardware infrastructure

All my model runs and inference results are captured in my [notebooks](../notebooks/).

You can see the GPU requirements (NVIDIA card name, VRAM size) for my experiments there.

Below is a collection of model and hardware test results, including mine.

## Guide on GPU requirements

How much GPU VRAM do I need to run the 7B model? Can 65B fit on 4x RTX 3090? Out of memory?

Here's a table showing a working combination of model sizes, GPU name, and VRAM
requirements. Some data points are reported by other users.

| Model  | MP* | GPU        | Min. VRAM  | Precision |Environment   | Throughput |
|--------|-----|------------|------------|-----------|--------------|------------|
| 7B     | 1   | A100 40GB  | 40GB       | TF32      | Colab Pro    |            |
| 7B^    | 1   | 3090 24GB  | 24GB       | FP32      | Home PC      |            |
| 7B     | 1   | A4000 16GB | 16GB       | TF32      | Home PC      |            |
| 7B**   | 1   | 3060 12GB  | 10GB       | int8      |              | 6-9it/s    |
| 7B**   | 1   | 3080 10GB  | 10GB       | int8      |              |            |
| 7B**   | 1   | 3090 24GB  | 10GB       | int8      |              | 30-50it/s  |
| 7B^^   | 1   | 2080 8GB   | 8GB        | FP32      | Home PC      |            |

| Model  | MP* | GPU        | Min. VRAM  | Precision |Environment   | Throughput |
|--------|-----|------------|------------|-----------|--------------|------------|
| 13B**  | 2   | 3090Ti 24GB| 20GB       | int8      |              | 10-30 it/s |
| 13B**  | 2   | 4090 24GB  | 20GB       | int8      |              | 10-30 it/s |

| Model  | MP* | GPU        | Min. VRAM  | Precision |Environment   | Throughput |
|--------|-----|------------|------------|-----------|--------------|------------|
| 33B**  | 4   | A6000 48GB | 40GB       | int8      |              | 20-40 it/s |
| 33B**  | 4   | A100 40GB  | 40GB       | int8      |              | 20-40 it/s |

| Model  | MP* | GPU        | Min. VRAM  | Precision |Environment   | Throughput |
|--------|-----|------------|------------|-----------|--------------|------------|
| 65B**  | 8   | A100 80GB  | 80GB       | int8      |              |            |

(_WIP: incomplete, accuracy is not fully confirm yet._)

_* Model Parallel (MP) encompasses both Pipeline Parallel (PP) and Tensor Parallel (TP)?_

_^ A modified of LLaMA model (`model.py`), configured for running with a single GPU (default is distributed GPU).Lowered batch size to 1 so the model can fit within VRAM._

_^^ A modified of LLaMA model. Only keep a single transformer block on the GPU at a time. Changed from fairscale layers to `torch.nn.Linear`. Details, see [this GitHub Issue](https://github.com/facebookresearch/llama/issues/79#issuecomment-1454707663)._

_** 8-bit quantized model._

### Memory requirements for each model size

Model arguments:
- `max_batch_size`: 2
- `max_seq_length`: 1024

| Model Params (Billions)  | 6.7  | 13   | 32.5 | 65.2 |
| ------------------------ | ---- | ---- | ---- | ---- |
| n_layers                 | 32   | 40   | 60   | 80   |
| n_heads                  | 32   | 40   | 52   | 64   |
| dimension                | 4096 | 5120 | 6656 | 8192 |

Memory requirements in 8-bit precision:

| Memory Requirements (GB) | 6.7  | 13   | 32.5 | 65.2 |
|--------------------------|------|------|------|------|
| Model (on disk)***       | 13   | 24   | 60   | 120  |
| Cache                    | 1    | 2    | 3    | 5    |
|                          |      |      |      |      |
| Total                    | 14   | 26   | 63   | 125  |

**Example:** 7B require at least 14GB VRAM in 8-bit (int8) precision. 28GB in 16-bit (fp16) precision (a doubled of VRAM).


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

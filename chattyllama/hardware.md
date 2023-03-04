# Hardware infrastructure

All my model runs and inference results are captured in my [notebooks](../notebooks/).

You can see the GPU requirements (NVIDIA card name, VRAM size) for my experiments there.

Below is a collection of model and hardware test results, including mine.

## Guide on GPU requirements

A table showing a working combination of model sizes, GPU name, and VRAM
requirements.

(_WIP: incomplete, accuracy is not fully confirmed yet._)

| Model  | MP* | GPU        | VRAM  | Precision |Environment   |
|--------|-----|------------|-------|-----------|--------------|
| 7B     | 1   | 1x A100    | 40GB  | 32-bit    | Colab Pro    |
| 7B^    | 1   | 1x RTX 3090| 24GB  | 32-bit    | Home PC      |
| 7B     | 1   | 1x A4000   | 16GB  | 32-bit    | Home PC      |
| 7B^^   | 1   | 1x RTX 2080| 8GB   | 32-bit    | Home PC      |

| Model  | MP* | GPU        | VRAM  | Precision |Environment   |
|--------|-----|------------|-------|-----------|--------------|
| 13B    | 2   |            |       |           |              |

| Model  | MP* | GPU        | VRAM  | Precision |Environment   |
|--------|-----|------------|-------|-----------|--------------|
| 33B    | 4   |            |       |           |              |

| Model  | MP* | GPU        | VRAM  | Precision |Environment   |
|--------|-----|------------|-------|-----------|--------------|
| 65B    | 8   |            |       |           |              |

_* Model Parallel (TP? or PP?)_

_^ A modified of LLaMA model (`model.py`), configured for running with a single GPU (default is distributed GPU).Lowered batch size to 1 so the model can fit within VRAM._

_^^ A modified of LLaMA model. Only keep a single transformer block on the GPU at a time. Changed from fairscale layers to `torch.nn.Linear`. Details, see [this GitHub Issue](https://github.com/facebookresearch/llama/issues/79#issuecomment-1454707663)._

_** 32-bit refers to TF32; mixed precision refers to Automatic Mixed Precision (AMP)._

### Memory requirements for each model size

Model arguments:
- `max_batch_size`: 2
- `max_seq_length`: 1024

| Model Params (Billions)  | 6.7  | 13   | 32.5 | 65.2 |
| ------------------------ | ---- | ---- | ---- | ---- |
| n_layers                 | 32   | 40   | 60   | 80   |
| n_heads                  | 32   | 40   | 52   | 64   |
| dimension                | 4096 | 5120 | 6656 | 8192 |

| Memory Requirements (GB) | 6.7  | 13   | 32.5 | 65.2 |
|--------------------------|------|------|------|------|
| Model (on disk)          | 13   | 25   | 65   | 130  |
| Cache                    | 1    | 2    | 3    | 5    |
|                          |      |      |      |      |
| Total                    | 14   | 27   | 68   | 135  |

[Spreadsheet](https://docs.google.com/spreadsheets/d/1EsRZcsvbITBCfb5N1toir5huomZxIq4lDNwQW3NfPRE/edit)
to calculate the memory requirements for each model size, following the FAQ and
paper. You can make a copy to adjust the batch size and sequence length.[^1]

Well, 65B needs a GPU cluster with a total of 135GB VRAM.

Some people just made enough code changes to **run the 7B model on the CPU**. [^2]

[^1]: [GitHub Issue: Post your hardware specs here if you got it to work](https://github.com/facebookresearch/llama/issues/79#issuecomment-1453814121)
[^2]: [GitHub Issue reply by "gmorenz"](https://github.com/facebookresearch/llama/issues/79#issuecomment-1454042028)
